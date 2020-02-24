## 홈페이지

> **이미지 분류(Image classification)**

1. 데이터 세트 선택
2. 뉴럴 네트워크 선택
　
 
> **한 에폭(1 epoch)**    

(epoch의 수는 전체 트레이닝 데이터 셋을 반복해서 학습하는 회수를 의미)
    
    
> **오버핏(overfit)**

다른 차이를 "학습"하는 대신, 트레이닝 데이터 셋의 어떤 이미지가 어떤 클래스가 속하는지 "암기"한다.
       
       
> **딥 러닝 워크플로우에는 두 가지 분명한 구성요소가 있습니다.**
1. 트레이닝/학습 (Training)
2. 구현 (Deployment)

> **객체탐지(Object Detection)**

딥 러닝에 대해 배우고 있는 지금, 우리는 아직까지 이미지분류를 위해 설계된 네트워크를 사용하는 방법만 알고 있습니다.
개체탐지 문제를 해결하는 첫번째 방법은 이미지분류네트워크를 기존의 프로그래밍과 결합하여 원하는 입력/출력쌍을 만드는 것입니다. 
이제 **"슬라이딩윈도우(Sliding Window)"** 라고 불리는 접근방식을 사용하겠습니다. 여기에서 이미지를 격자사각형이라고 불리는 작은 부분들로 나누어, 이미지분류기에 통과시키겠습니다.




----



## WEEK1

![ML역사 그림](https://github.com/lkeonwoo94/DL_cv-mdt_NVIDIA_Cert_Course_StudyPI/blob/master/ML.png)


http://solarisailab.com/deep-learning

```
* Computer Vision은 **영상**, 이미지 위주로만 다룬다 -> Classification, Detection, Segmentation.
* MDT는 Word generation과 영상분석을 포함하는 Multiple Data type.
* MDT는 2개의 data를 합쳐서 뭔가를 만들어낸다. 예를 들어 이미지가 주어지면 Captioning을 한다던가, 특징을 추출해서 binding.
* DLI는 이론 위주가 아닌, 다 안다는 전제하에 실습 위주로 한다.

* NIVIDA 제공의 DIGIT interface 사용.
* 딥러닝 엔진은 Tensorflow, pytorch 등등 존재, Caffe 기반으로 돌릴 예정 (플랫폼)
```


> **Backpropagation & Vanishing Gradient**

```
* backpropagation : a라는 실제를 train결과 b라는 예측을 했을 때, weight를 조절하는 것
* sigmoid -> gradien vanishing -> weight가 update가 안되요 (미분한게 0이라서)
* backpropagation은 layer의 입/출력에 대해 **각각이 서로 얼마나 영향을 미치는지**.
* loss function은 정답과의 차이를 어떻게 넘겨주겠다. (제곱으로 하겠다, 절대값으로 하겠다)
```
![vanishing](https://github.com/lkeonwoo94/DL_cv-mdt_NVIDIA_Cert_Course_StudyPI/blob/master/CV/img/underfit-vanishing.jpg)
    



> **IMAGE PROCESSING**     

 ###### **png가 제일 좋다.**

> **딥러닝 특징**
```
* ML와 비교했을 때 feature extraction이 따로 없음
* FC= Flatten
* labeling 매긴 것을 GT=Ground Truth

* 딥러닝이 뭘하냐? 정말 많은 feature를 뽑아요. feature 하나당 filter 하나 있다고 생각하시면 돼요.
* feature map을 뽑기 위해서, 계수들이 중요한데, 그 계수들을 하나하나 업데이트 해주는 거에요.
* 그러면서 feature가 layer를 지나면서 high level이 돼요.
* high level이 무슨 의미냐면, 앞에서는 low 한 feature (현미경같은). high level은 반대.

```

> **CNN**
```
* image map = feauture map
* RGB 이미지는 feature를 3개 뽑았다 (3 channel)
* 32x32x3 -> 28x28x6 이 되었다 ? 이게 **feature extraction** 됐다고 보시면 돼요.

* 1차원에서 Conv.는 훑고 지나가는것.
* 마찬가지로 2차원에서도 훑고 지나가는 컨셉을 이해하시면 돼요.
* Conv의 개념은 1D던,2D던 다 **Feature를 뽑아낸다**고 생각하시면 돼요.

* 학습이 된다는 의미는. 0과 1이라는 필터(filter == kernel)가 업데이트 되면서 조금씩 바뀌어요.
* 필터가 처음부터 끝까지 다 sliding을 해요.

* pixel 값이 0에 가까울 수록 검은색. 255에 가까울 수록 흰색.
* **흰색일 수록 feature가 뽑혔다고 생각하면 돼요**
```

> **Segmentation**
```
* Object detection은 box를 쳐주는데, box를 통해 위치정보를 알 수있죠.
* Semantic Segmentation은 pixel 하나하나씩 봐서 고양이인지 산인지 풀인지를 따지는 것. 각각의 요소가 뭔지 모르고 구분한다.
* Instance Segmentation은 개들을 빨간색으로 하면서, 서로 다른 개들을 segmentation 해주는거에요. label1, label2.
* Panoptic segmentation은 instance segmentation에서 하나 더 나아가서, 배경까지 segmentation 하는 것
```



> **Challenge**
* 이미지넷은 1000개 정도로 레이블링이 많이 되어있어요. 여기서 competition을 많이 합니다(challenge)

### 자잘한 Tip
```
* learning rate은 처음에 세게 줘야함 (local optima 빠지지 않기 위해서)
* 그래서 learning rate decay가 존재함.
```
```
* epoch -> data shuffling
* batch(=iter) & update(학습) ∈ epoch
```
```
* layer가 깊어질수록(feature map의 depth가 깊어질수록) resolution이 줄어들어요 (메모리가 줄어들어서)
* FC : 2차원이 1차원으로 바뀜 (max pooling하고 vector로 만듬) (공간정보 없어짐)
```

-----

## WEEK2
## [캐글](https://www.kaggle.com/)


![오버피팅](https://github.com/lkeonwoo94/DL_cv-mdt_NVIDIA_Cert_Course_StudyPI/blob/master/CV/img/overfit.png)   

> ## **overfitting**

* 데이터가 많은 경우: **class들간에 각각의 데이터 양이 일정하게 비슷해야** overfitting을 피할 수 있다.    
                      overfitting이 발생한다면, 많은 수의 data를 포함하는 class에 치우쳐져서 학습되기 때문입니다.     


* 데이터가 적은 경우: overfitting이 발생할 수 있다. 1장과 조금만 다른 영상이 입력으로 온다면 틀릴 가능성이 높습니다.    
                      **최소 클래스별 500개**정도씩 가지고 계시다면 시작해볼 순 있을 것 같습니다.    


* little overfitting : validation과 test가 비슷한 수치로 상승 => **model을 더 깊게 만들어서 해결**  /   **트레이닝을 너무 적게한 케이스**
* strong overfitting : valdiation과 test가 벌어짐 => **Regularization, Early stopping, Dropout 등으로 해결** / **이게 진짜 오버피팅이라고 할 수 있어요**

> ## **deployment**
학습된 모델을 porting 하는 것.

> 실습

1. GPU TASK2 : new dataset 생성
2. GPU TASK3 : deployment


---

# Transfer Learning
* 전처리 :  resize(train/test) , normalize(계산량을 줄임 & 값이 잘나옴)

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



