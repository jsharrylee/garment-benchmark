#!/usr/bin/env node

/**
 * Deterministically extract the named drafting geometry and production marks
 * emitted by FreeSewing Teagan 4.10.1.
 *
 * This is deliberately a cross-source adapter, not a reconstruction of
 * FreeSewing's creation-time operation trace.  It preserves FreeSewing's own
 * point/path/snippet names so they can be evaluated separately from the
 * GarmentCode runtime trace.
 */

import { createRequire } from 'node:module'
import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..', '..')
const RUNTIME_ROOT = path.join(REPO_ROOT, 'external', 'freesewing-tshirt-runtime')
const RUNTIME_PACKAGE = path.join(RUNTIME_ROOT, 'package.json')
const runtimeRequire = createRequire(RUNTIME_PACKAGE)

const ROUND_DIGITS = 6
const SCALE = 10 ** ROUND_DIGITS

function fail(message) {
  process.stderr.write(`extract_freesewing_teagan: ${message}\n`)
  process.exitCode = 2
}

function usage() {
  return [
    'Usage:',
    '  node benchmark/scripts/extract_freesewing_teagan.mjs [options]',
    '',
    'Options:',
    '  --model NAME               @freesewing/models export (default: cisFemaleAdult38)',
    '  --measurements-file PATH   JSON object in millimetres; overlays the selected model',
    '  --options JSON             Teagan option overrides as a JSON object',
    '  --options-file PATH        Teagan option overrides from a JSON file',
    '  --sa MM                    Seam allowance in millimetres (default: 10)',
    '  --output PATH              Write JSON to PATH instead of stdout',
    '  --list-models              Print supported adult test-model names',
    '  --help                     Show this help',
  ].join('\n')
}

function parseArgs(argv) {
  const args = {
    model: 'cisFemaleAdult38',
    measurementsFile: null,
    optionsJson: null,
    optionsFile: null,
    sa: 10,
    output: null,
    listModels: false,
    help: false,
  }
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    const take = () => {
      i += 1
      if (i >= argv.length) throw new Error(`missing value for ${arg}`)
      return argv[i]
    }
    if (arg === '--model') args.model = take()
    else if (arg === '--measurements-file') args.measurementsFile = take()
    else if (arg === '--options') args.optionsJson = take()
    else if (arg === '--options-file') args.optionsFile = take()
    else if (arg === '--sa') args.sa = Number(take())
    else if (arg === '--output') args.output = take()
    else if (arg === '--list-models') args.listModels = true
    else if (arg === '--help' || arg === '-h') args.help = true
    else throw new Error(`unknown argument: ${arg}`)
  }
  if (!Number.isFinite(args.sa) || args.sa < 0)
    throw new Error('--sa must be a finite, non-negative number in millimetres')
  if (args.optionsJson && args.optionsFile)
    throw new Error('use only one of --options or --options-file')
  return args
}

function sortedObject(entries) {
  return Object.fromEntries([...entries].sort(([a], [b]) => a.localeCompare(b)))
}

function normalizeNumber(value) {
  if (!Number.isFinite(value)) throw new Error(`non-finite geometry value: ${value}`)
  const rounded = Math.round(value * SCALE) / SCALE
  return Object.is(rounded, -0) || Math.abs(rounded) < 0.5 / SCALE ? 0 : rounded
}

function normalizeJson(value) {
  if (typeof value === 'number') return normalizeNumber(value)
  if (Array.isArray(value)) return value.map(normalizeJson)
  if (value && typeof value === 'object')
    return sortedObject(Object.entries(value).map(([key, item]) => [key, normalizeJson(item)]))
  return value
}

async function readJson(filePath, label) {
  let parsed
  try {
    parsed = JSON.parse(await readFile(path.resolve(filePath), 'utf8'))
  } catch (error) {
    throw new Error(`could not read ${label} JSON at ${filePath}: ${error.message}`)
  }
  if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object')
    throw new Error(`${label} JSON must be an object`)
  return parsed
}

async function importRuntimePackage(packageName) {
  const modulePath = runtimeRequire.resolve(packageName)
  return import(pathToFileURL(modulePath).href)
}

function loadPackageMetadata(packageName) {
  const packagePath = path.join(
    RUNTIME_ROOT,
    'node_modules',
    ...packageName.split('/'),
    'package.json'
  )
  return readFile(packagePath, 'utf8').then((text) => JSON.parse(text))
}

function attributesOf(entity) {
  return normalizeJson(entity?.attributes?.list ?? {})
}

function buildPointIdentityIndex(part) {
  const index = new Map()
  for (const name of Object.keys(part.points ?? {}).sort()) {
    const point = part.points[name]
    if (!index.has(point)) index.set(point, [])
    index.get(point).push(name)
  }
  return index
}

function encodePoint(point, identityIndex = null) {
  if (!point) return null
  const refs = identityIndex?.get(point) ?? []
  return {
    x_mm: normalizeNumber(point.x),
    y_mm: normalizeNumber(point.y),
    source_name: typeof point.name === 'string' ? point.name : null,
    point_refs: [...refs].sort(),
    attributes: attributesOf(point),
  }
}

function encodePath(pathValue, identityIndex) {
  const operations = []
  let current = null
  let subpathStart = null
  for (let index = 0; index < pathValue.ops.length; index += 1) {
    const op = pathValue.ops[index]
    const encoded = { index, type: op.type }
    if (op.id) encoded.id = op.id
    if (op.type === 'move') {
      encoded.from = null
      encoded.to = encodePoint(op.to, identityIndex)
      current = op.to
      subpathStart = op.to
    } else if (op.type === 'line') {
      encoded.from = encodePoint(current, identityIndex)
      encoded.to = encodePoint(op.to, identityIndex)
      current = op.to
    } else if (op.type === 'curve') {
      encoded.from = encodePoint(current, identityIndex)
      encoded.cp1 = encodePoint(op.cp1, identityIndex)
      encoded.cp2 = encodePoint(op.cp2, identityIndex)
      encoded.to = encodePoint(op.to, identityIndex)
      current = op.to
    } else if (op.type === 'close') {
      encoded.from = encodePoint(current, identityIndex)
      encoded.to = encodePoint(subpathStart, identityIndex)
      current = subpathStart
    }
    operations.push(encoded)
  }
  return {
    hidden: Boolean(pathValue.hidden),
    attributes: attributesOf(pathValue),
    length_mm: normalizeNumber(pathValue.length()),
    operations,
  }
}

function encodeSnippet(snippet, identityIndex) {
  return {
    type: snippet.def,
    anchor: encodePoint(snippet.anchor, identityIndex),
    attributes: attributesOf(snippet),
  }
}

function encodeCutlistEntry(entry) {
  if (!entry) return null
  const materials = sortedObject(
    Object.entries(entry.materials ?? {}).map(([material, cuts]) => [
      material,
      cuts.map((cut) => normalizeJson(cut)),
    ])
  )
  return {
    materials,
    cut_on_fold_segment:
      Array.isArray(entry.cutOnFold) && entry.cutOnFold.length === 2
        ? entry.cutOnFold.map((point) => encodePoint(point))
        : null,
    grain_origin: entry.grainOrigin ?? null,
    grain_degrees: typeof entry.grain === 'number' ? normalizeNumber(entry.grain) : null,
  }
}

function encodePart(part, cutlistEntry) {
  const identityIndex = buildPointIdentityIndex(part)
  const points = sortedObject(
    Object.entries(part.points ?? {}).map(([name, point]) => [name, encodePoint(point, identityIndex)])
  )
  const paths = sortedObject(
    Object.entries(part.paths ?? {}).map(([name, pathValue]) => [
      name,
      encodePath(pathValue, identityIndex),
    ])
  )
  const snippets = sortedObject(
    Object.entries(part.snippets ?? {}).map(([name, snippet]) => [
      name,
      encodeSnippet(snippet, identityIndex),
    ])
  )
  return {
    hidden: Boolean(part.hidden),
    points,
    paths,
    snippets,
    cutlist: encodeCutlistEntry(cutlistEntry),
  }
}

function semanticPoint(parts, partName, sourcePointName, evidence, note = null) {
  const point = parts[partName]?.points?.[sourcePointName]
  if (!point)
    return {
      status: 'absent',
      part: partName,
      source_point_name: sourcePointName,
      evidence,
      note,
      coordinate: null,
    }
  return {
    status: 'present',
    part: partName,
    source_point_name: sourcePointName,
    evidence,
    note,
    coordinate: encodePoint(point),
  }
}

function canonicalSemantics(rawParts) {
  const exact = 'exact_author_named_point'
  return {
    landmarks: {
      FNP: [semanticPoint(rawParts, 'teagan.front', 'cfNeck', exact, 'front neck at center fold')],
      BNP: [semanticPoint(rawParts, 'teagan.back', 'cbNeck', exact, 'back neck at center fold')],
      SNP: [
        semanticPoint(rawParts, 'teagan.front', 'neck', exact, 'front side-neck point'),
        semanticPoint(rawParts, 'teagan.back', 'neck', exact, 'back side-neck point'),
      ],
      SP: [
        semanticPoint(rawParts, 'teagan.front', 'shoulder', exact, 'front shoulder endpoint'),
        semanticPoint(rawParts, 'teagan.back', 'shoulder', exact, 'back shoulder endpoint'),
      ],
      BP: {
        status: 'absent_by_recipe',
        evidence: 'Teagan does not create or consume a bust-point landmark for this T-shirt draft',
        coordinates: [],
      },
    },
    horizontal_levels: {
      BL: {
        status: 'approximate_proxy',
        meaning: 'FreeSewing Teagan chest-line level, used only as a bust-line proxy',
        source_measurement: 'hpsToBust',
        instances: [
          semanticPoint(rawParts, 'teagan.front', 'cbChest', 'approximate_author_named_point'),
          semanticPoint(rawParts, 'teagan.back', 'cbChest', 'approximate_author_named_point'),
        ],
        rendered_source_path: 'chest',
      },
      WL: {
        status: 'explicit',
        meaning: 'waist level from high-point shoulder',
        source_measurement: 'hpsToWaistBack',
        instances: [
          semanticPoint(rawParts, 'teagan.front', 'cbWaist', exact),
          semanticPoint(rawParts, 'teagan.back', 'cbWaist', exact),
        ],
        rendered_source_path: 'waist',
      },
      HL: {
        status: 'explicit_named_construction_level_not_rendered_as_guide',
        meaning: 'hips level',
        source_measurement: 'waistToHips',
        instances: [
          semanticPoint(rawParts, 'teagan.front', 'cbHips', exact),
          semanticPoint(rawParts, 'teagan.back', 'cbHips', exact),
        ],
        rendered_source_path: null,
      },
    },
    darts: {
      status: 'absent_by_design',
      applicable: false,
      operations: [],
    },
  }
}

function productionSemantics(rawParts, encodedParts, saMm) {
  const notches = []
  const grainlines = []
  const cutOnFold = []
  const seamAllowancePaths = []
  for (const partName of Object.keys(rawParts).sort()) {
    // Inherited FreeSewing source parts remain useful in `parts`, but their
    // annotations are duplicated by the visible Teagan parts.  Production
    // mark counts therefore use only the final visible pattern pieces.
    if (rawParts[partName].hidden) continue
    const encoded = encodedParts[partName]
    for (const [snippetName, snippet] of Object.entries(encoded.snippets)) {
      if (snippet.type === 'notch' || snippet.type === 'bnotch')
        notches.push({
          part: partName,
          snippet_name: snippetName,
          notch_type: snippet.type,
          anchor: snippet.anchor,
        })
    }
    for (const pathName of Object.keys(encoded.paths)) {
      if (pathName.includes('__macro_grainline_'))
        grainlines.push({ part: partName, path_name: pathName, path: encoded.paths[pathName] })
      if (pathName.includes('__macro_cutonfold_'))
        cutOnFold.push({ part: partName, path_name: pathName, path: encoded.paths[pathName] })
      if (pathName === 'sa')
        seamAllowancePaths.push({ part: partName, path_name: pathName, path: encoded.paths[pathName] })
    }
  }
  return {
    notches: {
      status: notches.length ? 'present' : 'absent_by_design',
      items: notches,
      note: 'Teagan 4.10.1 emits matching front/back armhole notches on bodice and sleeve.',
    },
    grainlines: {
      status: grainlines.length ? 'present' : 'absent',
      items: grainlines,
      note: 'Front/back use a combined cut-on-fold and grainline annotation; sleeve uses an explicit grainline.',
    },
    cut_on_fold: {
      status: cutOnFold.length ? 'present' : 'absent',
      items: cutOnFold,
    },
    seam_allowance: {
      status: saMm > 0 ? 'present' : 'disabled_by_input',
      requested_mm: normalizeNumber(saMm),
      paths: seamAllowancePaths,
      source_policy: {
        front_and_back: 'hem = 3 * sa; side, armhole, shoulder, and neckline = 1 * sa; fold edge is not offset',
        sleeve: 'hem = 3 * sa; remaining boundary = 1 * sa',
      },
    },
  }
}

async function main() {
  let args
  try {
    args = parseArgs(process.argv.slice(2))
  } catch (error) {
    fail(`${error.message}\n\n${usage()}`)
    return
  }
  if (args.help) {
    process.stdout.write(`${usage()}\n`)
    return
  }

  let teaganModule
  let modelsModule
  try {
    ;[teaganModule, modelsModule] = await Promise.all([
      importRuntimePackage('@freesewing/teagan'),
      importRuntimePackage('@freesewing/models'),
    ])
  } catch (error) {
    fail(
      `the ignored runtime is unavailable at ${RUNTIME_ROOT}; install nothing automatically. ` +
        `Expected the pinned local packages. Original error: ${error.message}`
    )
    return
  }

  const supportedModels = Object.keys(modelsModule)
    .filter((name) => /^cis(?:Female|Male)Adult\d+$/.test(name))
    .sort()
  if (args.listModels) {
    process.stdout.write(`${supportedModels.join('\n')}\n`)
    return
  }
  if (!supportedModels.includes(args.model)) {
    fail(`unsupported model ${args.model}; choose one of: ${supportedModels.join(', ')}`)
    return
  }

  try {
    const baseMeasurements = normalizeJson(modelsModule[args.model])
    const measurementOverrides = args.measurementsFile
      ? normalizeJson(await readJson(args.measurementsFile, 'measurements'))
      : {}
    const requestedMeasurements = { ...baseMeasurements, ...measurementOverrides }
    let requestedOptions = {}
    if (args.optionsJson) requestedOptions = JSON.parse(args.optionsJson)
    if (args.optionsFile) requestedOptions = await readJson(args.optionsFile, 'options')
    if (!requestedOptions || Array.isArray(requestedOptions) || typeof requestedOptions !== 'object')
      throw new Error('options must be a JSON object')

    // Constructor arguments are FreeSewing setting sets themselves; settings
    // must not be nested under a `settings` key.
    const pattern = new teaganModule.Teagan({
      measurements: requestedMeasurements,
      options: requestedOptions,
      complete: true,
      sa: args.sa,
      units: 'metric',
      layout: false,
    }).draft()
    const rawParts = pattern.parts?.[0]
    if (!rawParts || !rawParts['teagan.front'] || !rawParts['teagan.back'] || !rawParts['teagan.sleeve'])
      throw new Error('Teagan draft did not return the expected front/back/sleeve parts in parts[0]')

    const cutlist = pattern.setStores?.[0]?.cutlist ?? {}
    const encodedParts = sortedObject(
      Object.entries(rawParts).map(([name, partValue]) => [
        name,
        encodePart(partValue, cutlist[name]),
      ])
    )
    const [teaganPackage, corePackage, modelsPackage] = await Promise.all([
      loadPackageMetadata('@freesewing/teagan'),
      loadPackageMetadata('@freesewing/core'),
      loadPackageMetadata('@freesewing/models'),
    ])
    const resolvedSettings = pattern.settings[0]
    const output = {
      schema_version: 'freesewing-teagan-extract/v1',
      source: {
        design: 'Teagan T-shirt',
        design_package: '@freesewing/teagan',
        design_version: teaganPackage.version,
        core_version: corePackage.version,
        models_version: modelsPackage.version,
        repository: 'https://codeberg.org/freesewing/freesewing',
        documentation: 'https://freesewing.eu/docs/designs/teagan/',
        source_code_license_spdx: teaganPackage.license,
        extraction_kind: 'post_draft_author_named_geometry',
        limitation:
          'This adapter preserves author-named points, paths, snippets, and cutlist data. It is not a creation-time operation trace or an expert validation of Korean drafting semantics.',
      },
      input: {
        model: args.model,
        requested_measurements_mm: normalizeJson(requestedMeasurements),
        resolved_measurements_mm: normalizeJson(resolvedSettings.measurements),
        requested_options: normalizeJson(requestedOptions),
        resolved_options: normalizeJson(resolvedSettings.options),
        resolved_absolute_options_mm: normalizeJson(resolvedSettings.absoluteOptions),
        seam_allowance_mm: normalizeNumber(args.sa),
        complete: true,
      },
      canonical_semantics: canonicalSemantics(rawParts),
      production_semantics: productionSemantics(rawParts, encodedParts, args.sa),
      parts: encodedParts,
    }

    const text = `${JSON.stringify(normalizeJson(output), null, 2)}\n`
    if (args.output) await writeFile(path.resolve(args.output), text, 'utf8')
    else process.stdout.write(text)
  } catch (error) {
    fail(error.stack ?? error.message)
  }
}

await main()
