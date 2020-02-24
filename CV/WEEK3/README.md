# Transfer Learning

목적 : 충분한 양의 데이터가 없을 때 (labeling도 되지 않은 경우, supervised learning이 필요할 때) 사용한다.
transfer learning을 하지 않으면 scratch learning이라고 한다. (바닥부터 시작 -> ex. 알파고)
차이는 수1 배운 애 한테 수2를 가르치는 것과, 배운적이 없는 학생에게 수2를 가르치는 것과 같다.
(minimize 된 weight를 가져옴)

저차원은 data가 많아도 뽑을 수 있는 feature의 특징이 한정적임.
그래서 고차원(뒷단)만 풀어서 사용한다. => 저차원을 freeze

만약 데이터가 많다 하면 high level feature의 freeze를 풀면 모델 성능이 올라간다.

다시한번 목적을 생각했을 때, data가 많으면 transfer learning은 필요가 없다. converge 되기 때문.
의료쪽에서는 20만장 이상 learning 하면 big data로 친다.

> 실습 4 생략
---

# Obj detect
1. sliding window classifier : 전 영역을 움직이면서 처리 (feature를 뽑고 class 인지 판단) (고전방법) -> candidate region 출력 
2. NMS : 1개의 box만 추출
3. Haar : 특정 객체를 잘 뽑은 filter가 있음. 그것을 sliding window 마다 연산함 (ex.  eye-detector , nose-detector)
3. HOG : 근접한 pixel의 map을 만들어서 미분값들의 차이 
4. DPM: HOG + SVM
5. Selective Search : ① pixel 값들의 인접한 차이로 segmentation을 진행  -> ② layer를 지나면서 rough한 candidate region을 추출
(고전방법)

OBJ 공개 데이터 셋 : 이미지넷 , 파스칼VOC, MSCOCO 데이터셋, 시티전경(시티스케이프)

# CNN 사용한 방법
> 2-stage
R-CNN 부터. : (피쳐맵 추출 - proposal)(한스테이지) - 반복
앵커 : 피쳐맵을 뽑기 위해 다양한 필터를 사용한다.

> 1-stage
grid가 촘촘할 수록 작은 영역을 추출한다.
레티나넷 : 피쳐맵이 작아지수록 찾아내는 obj의 크기가 커진다.

---
# 실습 task5

filter size 만큼 뜯어서 cnn 넣고 classifier 돌리면, size에 대해서 이것이 무엇인지에 대한 확률(confidence 값)이 hitmap으로 나옴.    
확률이 높은 영역을 box를 치면, detection이 됨. 
-> stride를 dense하게 했을 때 변화를 보자

> 두번째 실습 방법

Fully Connected Layer 사용. (FC = Flatten)    

Deconv / Upsampling으로 Output을 키운다( FC대신 Fully convolution.)    

> 세번째 접근법 (DetectNet : NVIDIA에서 개발한 방법)

feature의 위치정보도 나와있음. detectuib 알고리즘은 **좌표정보**도 필요. 좌표정보도 loss로 들어가서 학습이 돼야함.    

재학습을 할때는 loss를 낮게 주는게 중요하다.    
B-box는 GT와 PD 사이의 거리를 통해서 학습을 시킨다.    
그래서 두개를 더한 loss가 줄어들게끔 학습을 합니다.    

GT는 정답을 직접 만들어야 한다. online에 label을 위한 툴이 있다. b-box를 넣으면 text file로 떨어진다.    
   
NMS를 쓰면 threshold에 따라서 b-box가 1개로 만들어짐.    

http://ec2-3-136-108-117.us-east-2.compute.amazonaws.com/1ZbODU8T/notebooks/tasks/task-assessment/task

Job Directory 복사.
