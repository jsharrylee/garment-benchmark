# 공개 주장 범위

## 현재 근거로 말할 수 있는 것

- GCDv2 로컬 3,450벌을 13,800장의 중성 4-view RGB와 analytic sewing-pattern 표현으로 연결했다.
- 원본 pattern → DSL → decoded pattern의 기하 round-trip을 검사했다.
- 2,995벌·11,980 view의 pixel-aligned visible-panel instance mask를 생성했다.
- 패널 사이를 ignore하던 target을 majority panel ID로 바꾸고 동일한 새 정답에서 mask A/B를 수행했다.
- 내부 단일-seed 비교에서 전체 matched-panel IoU가 0.6003→0.6813, inter-panel boundary IoU가 0.4475→0.5212로 개선됐다.
- 같은 17,744 train / 2,027 selection 패널에서 line-configuration top-1이 50.27%→52.74%, top-10 coverage가 92.25%→93.04%, reranked top-1이 56.64%→58.95%로 개선됐다.
- train-only panel bank가 validation canonical line structure의 99.89%를 oracle 기준으로 포함했다.
- 최신 reranker의 covered error 691건 중 57.60%는 L/Q/C/A edit distance 2 이하였고, 72.94%는 edge count가 같거나 ±1이었다.

## 수치와 함께 반드시 붙여야 하는 제한

- 최신 결과는 seed 17 한 번의 내부 selection 진단이다.
- panel query와 source panel의 대응에는 GT mask 기반 Hungarian matching을 사용했다.
- pretraining checkpoint selection과 downstream 독립 평가가 완전히 분리된 공식 test 결과가 아니다.
- line configuration은 cycle 시작점과 traversal reversal에 불변인 `edge count + L/Q/C/A sequence`다.
- mask IoU 개선과 panel-count MAE는 같은 방향으로 움직이지 않았다. panel-count MAE는 1.5584→1.7727로 악화됐다.
- 과거 IoU 약 0.86은 회색 ignore가 포함된 옛 target에서 계산했으므로 새 target의 0.6813과 직접 비교하지 않는다.

## 주장하면 안 되는 것

- 사진에서 편집 가능한 완성 2D pattern을 복원했다.
- 모델이 패널 선의 좌표·길이·각도·곡률·제어점을 예측했다.
- line-configuration top-1 58.95%가 garment-level 성공률이다.
- 모델이 source edge의 semantic role이나 seam partner를 맞혔다.
- 출력 패턴의 봉제 가능성, 핏 또는 CLO simulation 성공을 검증했다.
- 실제 사진, 다른 생성기 또는 외부 패턴으로 일반화했다.
- 여러 seed와 untouched test에서 재현된 확정 성능이다.

## 권장 한 문장

> 4-view garment render와 analytic sewing-pattern graph를 연결하고, visible-panel segmentation과 canonical boundary-primitive classification을 단계적으로 학습했다. 패널 경계를 supervision에 포함하자 동일 내부 집단에서 mask boundary와 L/Q/C/A 구성 선택이 함께 개선됐지만, 연속 geometry·seam·완성 패턴 복원은 아직 미해결이다.
