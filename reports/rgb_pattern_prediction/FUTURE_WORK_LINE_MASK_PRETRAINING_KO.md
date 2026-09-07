# Future Work — source-edge mask를 이용한 line-aware ViT pretraining

상태: 설계안, 아직 데이터 전량 생성·학습하지 않음  
선행 조건: 새 데이터 계약과 split을 파일로 고정하고 pilot gate를 통과할 것

## 1. 왜 필요한가

현재 panel-mask pretraining은 다음 질문을 가르친다.

> 이 픽셀들은 같은 패널에 속하는가?

하지만 line-configuration retrieval에 필요한 질문은 더 세밀하다.

> 이 패널 윤곽은 어디에서 서로 다른 source edge로 나뉘며, 각 구간은 직선·quadratic Bézier·cubic Bézier·원호 중 무엇인가?

최신 v4.4에서는 정답 구조가 top-10에 93.04% 남았지만 최종 top-1은 58.95%였다. 정답이 top-10에 있는데도 실패한 691건 중 57.60%는 최종 선택과 정답의 edit distance가 2 이하였고, 72.94%는 edge count가 같거나 ±1이었다. 대표 혼동은 `LALA ↔ LLLL`, `LLLCC ↔ LLLC`였다.

따라서 panel 면적만 가르치는 것보다 **source edge의 위치, 경계 분할점, endpoint, view별 가시성**을 직접 가르치는 것이 현재 병목과 더 잘 맞는다.

## 2. 기존 heatmap을 그대로 쓰지 않는 이유

현재 mask 데이터의 neckline·armhole·hem heatmap은 시각화 후보일 뿐이다.

- 모든 `*_valid.png`가 0이다.
- `boundary_supervision_approved=false`다.
- 연결된 mesh chain 하나를 찾았다는 것만 확인했으며, 원본 source edge 전체를 빠짐없이 덮었는지는 입증하지 않았다.
- hem 등 일부 label은 규칙 기반 후보라 모든 garment type을 포괄하지 않는다.

따라서 기존 heatmap의 valid flag를 바꾸지 않는다. 새 `line-mask/v1` 데이터 계약과 새 출력 디렉터리를 만든다.

## 3. 한 view의 새 정답 묶음

파일 수가 폭증하지 않도록 production 데이터는 edge마다 PNG 한 장을 만들지 않고, 하나의 instance raster와 table을 사용한다. 개별 edge mask는 이 둘에서 즉시 복원할 수 있다.

```text
sample_id/view/
├── rgb.png
├── panel_instance_u16.png
├── panel_valid_u8.png
├── edge_instance_u16.png
├── edge_valid_u8.png
├── boundary_distance_f16.npz
├── endpoint_heatmaps_f16.npz
└── edge_table.json
```

### `edge_instance_u16.png`

- 0은 background, 1 이상은 해당 view 안의 visible source-edge instance ID다.
- 같은 source edge라도 다른 의상에서 같은 ID를 보장하지 않는다.
- `edge_table.json`이 `(source panel ID, source edge index)`에 연결한다.

### `edge_valid_u8.png`

- 가려짐, self-occlusion, mapping 불확실, rasterization 충돌 주변은 0으로 둔다.
- 보이지 않는 edge를 `없음`이라는 음성 정답으로 학습시키지 않는다.

### `boundary_distance_f16.npz`

- 1-pixel line은 작은 camera 차이에 지나치게 민감하므로 source edge 중심선까지의 distance 또는 Gaussian heatmap을 저장한다.
- 실제 학습 폭은 config에서 정하고 원본 중심선 좌표는 별도 table에 남긴다.

### `endpoint_heatmaps_f16.npz`

- 인접 edge가 만나는 source endpoint를 view에 투영한 heatmap이다.
- 모델이 패널 외곽선만 찾고 edge 분할점을 무시하는 것을 막는 핵심 보조 정답이다.

### `edge_table.json`

각 edge마다 다음을 저장한다.

- sample-local edge instance ID
- source panel ID와 source edge index
- primitive type `L/Q/C/A`
- source endpoints와 active control/arc parameter의 참조
- 대응하는 mesh edge chain
- view별 projected polyline과 visible fraction
- endpoint visibility
- mapping/coverage/rasterization valid flag와 실패 사유
- seam에 사용되는 edge인지 여부. 단, seam role 자체를 visibility 정답과 혼동하지 않는다.

## 4. source edge를 3D RGB 위치로 옮기는 과정

1. 원본 2D pattern에서 `(panel, edge index)`와 analytic curve를 읽는다.
2. source panel의 경계가 simulation mesh에서 어떤 vertex/edge chain으로 전달됐는지 기존 topology·stitch 대응 기록으로 추적한다.
3. chain의 방향과 양 끝을 원본 edge 방향에 맞춘다.
4. RGB와 동일한 카메라로 3D chain을 투영한다.
5. RGB 생성 때 사용한 depth buffer로 다른 삼각형 뒤에 가려진 구간을 제거한다.
6. visible polyline을 instance raster, distance heatmap, endpoint heatmap으로 저장한다.
7. 원본 source edge coverage와 raster 결과를 검사한 뒤 valid flag를 결정한다.

봉제된 edge는 simulation mesh에서 열린 외곽 경계가 아닐 수 있다. 따라서 단순히 mesh의 boundary edge를 찾는 방식으로 만들면 안 되고, source-to-mesh 대응을 사용해야 한다.

## 5. bulk 생성 전에 통과해야 할 mapping gate

먼저 의상 종류별 20벌씩, 총 100벌의 stratified pilot을 만든다. 다음 검사를 파일로 고정하고, threshold는 **bulk 실행 전에** 확정한다.

- source edge마다 대응 mesh chain이 하나의 비분기 연속 경로인가
- chain 양 끝이 source endpoint correspondence와 일치하는가
- chain이 해당 source curve 전체를 덮고 일부만 선택한 것은 아닌가
- 한 패널의 서로 다른 source edge가 같은 mesh segment를 중복 소유하지 않는가
- sewn edge와 free boundary 모두 올바르게 추적되는가
- 같은 camera/depth에서 occlusion 판정이 RGB와 픽셀 정렬되는가
- 너무 짧거나 subpixel인 edge를 valid/invalid 중 어떻게 처리했는가
- L/Q/C/A별, garment category별, 앞·뒤·옆 view별 coverage 통계가 남는가
- contact sheet의 instance ID, endpoint, source SVG가 사람이 보아도 일치하는가

실패한 edge나 view는 자동 보정해 통과시키지 않고 사유와 함께 invalid로 남긴다. 한 의상의 일부 edge가 invalid라고 해서 다른 valid edge까지 반드시 버릴 필요는 없지만, loss mask가 이를 정확히 반영해야 한다.

## 6. 학습 구조

초기 가설은 panel decoder를 버리는 것이 아니라 계층적으로 확장하는 것이다.

```text
4-view neutral RGB
  ↓
ImageNet ViT-B/16
  ↓
dense visual tokens
  ├── panel queries → panel masks / panel visibility
  └── panel-conditioned edge queries
        ├── visible source-edge mask 또는 centerline heatmap
        ├── endpoint heatmaps
        ├── edge visibility
        └── optional L/Q/C/A head
```

edge query는 먼저 대응된 panel query의 영역 안에서 작동하도록 한다. edge instance는 순서 없는 set으로 match하되, 같은 panel에 속한 edge들의 endpoint 연결이 하나의 cycle을 이루는지 보조 consistency loss를 둔다.

예상 loss는 다음 구성이다.

```text
L = L_panel_object
  + λ1 L_panel_mask
  + λ2 L_edge_object
  + λ3 L_edge_heatmap
  + λ4 L_endpoint
  + λ5 L_visibility
  + λ6 L_cycle_consistency
  + λ7 L_primitive_type(optional)
```

`primitive_type`은 보조 head로만 시작한다. 3D에 투영된 polyline만으로 quadratic, cubic, arc를 항상 구별할 수 있는 것은 아니기 때문이다. 먼저 edge 위치와 분할점을 배우는 효과와 primitive type 분류 효과를 따로 평가한다.

## 7. 데이터 split과 누수 방지

- line-mask artifact는 가능한 범위 전체에 생성할 수 있지만 pretraining gradient는 effective-training에서만 계산한다.
- checkpoint 선택도 effective-training 내부의 고정 selection만 사용한다.
- historical v1처럼 downstream validation ID를 pretraining checkpoint 선택에 사용하지 않는다.
- 같은 source pattern의 duplicate와 파생본은 split을 넘지 않도록 group key로 묶는다.
- official test 165벌은 preregistered multi-seed gate를 통과하기 전까지 image, feature, prediction 평가를 하지 않는다.
- 데이터 계약, ID digest, code hash, camera hash, source mapping version을 run 전에 잠근다.

## 8. 효과를 어떻게 분리해서 확인할까

동일한 downstream 구조와 학습 조건으로 세 encoder를 비교한다.

| Arm | Encoder |
|---|---|
| A | generic ImageNet ViT |
| B | clean train-only panel-mask-pretrained ViT |
| C | 같은 split에서 panel+line-mask-pretrained ViT |

모든 encoder를 frozen으로 두고 같은 초기값의 downstream model을 새로 학습한다.

### Pretraining 자체 지표

- panel matched IoU/Dice와 count MAE
- visible source-edge mask/centerline F1
- endpoint localization error 또는 PCK
- edge count MAE
- mapping-valid edge에 한정한 L/Q/C/A accuracy
- primitive, category, view, visible-length 구간별 성능

### Downstream 1차 지표

- v4.2 canonical line-configuration top-1/top-5/top-10
- macro recall 및 rare-frequency bucket recall
- top-10 후보 coverage를 유지하면서 top-1이 개선되는지

### Downstream 2차 지표

- v4.4 candidate-conditioned reranker overall/conditional top-1
- 대표 혼동 `LALA↔LLLL`, `LLLCC↔LLLC` 감소
- panel·edge count, continuous geometry, seam은 보조 지표

성공 기준은 실행 전에 별도 run spec에 수치로 고정한다. overall top-1만 오르고 macro recall이나 rare bucket이 악화되는 경우를 성공으로 보지 않는다. seed 17에서 방향을 보고, 사전등록 gate 통과 시에만 공통 seed를 추가한다.

## 9. 이 실험으로도 해결되지 않는 것

- 보이지 않는 back panel의 정확한 topology와 geometry
- 시각적으로 같은 contour를 만드는 서로 다른 analytic primitive 표현의 비식별성
- 실제 사진의 texture, pose, body-shape, lighting domain shift
- seam partner와 봉제 가능성
- 정확한 실제 cm 크기. camera ortho scale을 입력으로 쓸지 별도 계약이 필요하다.

line-aware pretraining의 직접 목적은 완성 패턴 생성이 아니라 **visible panel boundary의 source-edge 분할과 line configuration 검색을 개선하는 것**이다.
