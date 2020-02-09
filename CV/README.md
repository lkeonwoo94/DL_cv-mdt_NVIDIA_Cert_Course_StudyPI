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
(8:00pm-9:00pm)

http://solarisailab.com/deep-learning


p1.
* Computer Vision은 **영상**, 이미지 위주로만 다룬다 -> Classification, Detection, Segmentation.
* MDT는 Word generation과 영상분석을 포함하는 Multiple Data type.
* MDT는 2개의 data를 합쳐서 뭔가를 만들어낸다. 예를 들어 이미지가 주어지면 Captioning을 한다던가, 특징을 추출해서 binding.

DLI는 이론 위주가 아닌, 다 안다는 전제하에 실습 위주로 한다.

1주차 정도는 이론 강의 하고, 2주차도 앞에 실습 뭘 다룰지 30분 정도 함.

쿠폰 등록은 계정에서 1년정도 유효함.끝까지 완주해서 Certification 취득하지면 될 거 같아요.

NVIDIA DLI는 AWS에 돈을 내고 거기에 Instance를 사는 겁니다. 이미 NVIDIA 컨텐츠와 GPU는 AWS를 쓰고 있습니다.
우리는 크롬만 들고 딥러닝을 활용할 예정입니다. 클라우드 상에서 다 한다.

p2.
Deployment : 어떻게 코딩 할 것인지
Performance : 실제 적용시 어떻게 퍼포먼스 높일지 ( 하이퍼파라미터 조절)

p3.
이미지 프로세싱 : digitized image

NIVIDA 제공의 DIGIT interface 사용할 것입니다. 
딥러닝 엔진은 Tensorflow, pytorch 이런게 있는데, Caffe 기반으로 돌릴 예정입니다.(플랫폼)


p4. 역사
2장.

모델 -> 학습을 서포트 해주는 GPU 개발 -> 빅데이터 커짐
조합이 잘 되서 딥러닝이 등장.


p7.
FC= Flatten
backpropagation : a라는 실제를 train결과 b라는 예측을 했을 때, weight를 조절하는 것
sigmoid -> gradien vanishing -> weight가 update가 안되요 (미분한게 0이라서)

p9.
왜 GPU를 많이 쓰냐면, 컴퓨터가 렌더링을 해서 모니터에 뿌려줘야합니다.
그럴 경우 단순 연산이 필요합니다. GPU는 단순연산 많이해서.
인공지능은 곱셈과 덧셈이 굉장히 많아요. 그래서 GPU의 힘을 빌리고, 병렬연산이 가능한 것도 쓰죠.

p10.
이미지넷은 1000개 정도로 레이블링이 많이 되어있어요. 여기서 competition을 많이 합니다(challenge)
2012년에 딥러닝을 사용하면서 에러가 10% 떨어졌죠.
labeling 매긴 것을 GT=Ground Truth



p13.
딥러닝은 처음에 simple한 feature 뽑고, 나중에는 complex 뽑음.
(ML와 비교했을 때 feature extraction이 따로 없음)

p15.
**CNN**
image map = feauture map
1. feature를 3개 뽑았어요 (3 channel)
점점 feature를 complex하게 뽑는다고 생각하시면 돼요.

p16.
Semantic Segmentation은 pixel 하나하나씩 봐서 고양이인지 산인지 풀인지를 따는거에요.
Semantic Segmentation은 각각의 요소가 뭔지 모르고 구분하는 것. (test set에서)
photoshop이 이게 굉장히 잘 되어있죠.

Object detection은 box를 쳐주는데, box를 통해 어디에 있는지 알 수있죠.

Instance Segmentation은 개들을 빨간색으로 하면서, 서로 다른 개들을 segmentation 해주는거에요. label1, label2.


이 수업에서는 segmentation와 detection까지 나갈 거에요.


p20.
gan은 noise를 주면 만들어내요


p24.
IMAGE PROCESSING
jpeg으로는 왠만하면 학습 안시키는게 좋아요(압축하면서 데이터 손실).
갤러리에서 사진 클릭하면 압축이 풀리면서 보여요.(디코딩) 저장시 이미 원본 데이터가 아니에요.
**png가 제일 좋다.**
dcm은 의료용
bmp로 하면 압축이 거의 안되고(simple한 압축 algorithm), png는 효율적으로 함.

p.28
1차원에서 Conv.는 훑고 지나가는것.
마찬가지로 2차원에서도 훑고 지나가는 컨셉을 이해하시면 돼요.

학습이 된다는 의미는. 0과 1이라는 필터(filter == kernel)가 업데이트 되면서 조금씩 바뀌어요.
필터가 처음부터 끝까지 다 sliding을 해요.
Conv의 개념은 1D던,2D던 다 **Feature를 뽑아낸다**고 생각하시면 돼요.


p30.
소벨 필터 보면 피쳐를 뽑고 붙여넣고.
pixel 값이 0에 가까울 수록 검은색. 255에 가까울 수록 흰색.
**흰색일 수록 feature가 뽑혔다고 생각하면 돼요**
딥러닝이 뭘하냐? 정말 많은 feature를 뽑아요. feature 하나당 filter 하나 있다고 생각하시면 돼요.
feature map을 뽑기 위해서, 계수들이 중요한데, 그 계수들을 하나하나 업데이트 해주는 거에요.
그러면서 feature가 layer를 지나면서 high level이 돼요.
high level이 무슨 의미냐면, 앞에서는 low 한 feature (현미경같은). high level은 반대.
기존 ML에서는 필터의 계수를 수학적으로 모델링 했는데, 지금은 몇x몇 와꾸만 잡아주면 알아서 튜닝이 되면서 바뀌어요.
Conv를 통해서, feature를 뽑기도 하고, 원본 영상을 blurring을 하기도 하고 합니다.

filter 내 weight의 sum은 0이 아니에요. (sobel filter의 sum은 0이길래 여쭤봣습니다.) (가우시안은 sum을 1로 맞추고요)
보통 edge를 추출하는 filter의 경우 sum이 0입니다.
sum이 0이 되어야 edge가 아닌 부분이 검게 나온다.

p32.
filter는 hyperparameter 들 중 하나이고, 뭐로 하던간에 잘돼요. hypterparameter는 실험 중에 바꿀 수 있어요.


p33.
backpropagation은 layer의 입/출력에 대해 **각각이 서로 얼마나 영향을 미치는지**.


p36.
loss function은 정답과의 차이를 어떻게 넘겨주겠다. (제곱으로 하겠다, 절대값으로 하겠다)


p38.
depth 3 = 3 channel = RGB (일반적)
32x32x3 -> 28x28x6 이 되었다 ? 이게 **feature extraction** 됐다고 보시면 돼요.
feature map은 생각보다 깨끗하지 않아요. noise도 있고요, 0 이하의 값들은 다 0으로 맞춰요(ReLU)


---

(9:00pm-10:00pm)
실습
loss 를 minimize 하는게 학습이 되는 원리.
영상 자체를 외워버리는게 overfit

[Open DIGITS](http://ec2-3-133-104-8.us-east-2.compute.amazonaws.com/g2yGjI5Q/datasets/20171102-180326-8901)

digit에서 imageset을 만들 수 있어요. 폴더의 링크만 넣어주면 알아서 labeling 돼요.
dataset 클릭해서 classification 문제인지 이런 task를 지정할 수 있어요.

explore the db를 클릭하면 이미 label된 그림을 볼 수 있어요.

epoch이 한 영상 을 다 한번 보여줬을 때
customize 누르면 model 구체적으로 코드를 볼 수도 있고, 시각화도 볼 수 있음.(연결이 잘 되었는지 확인)
이거는 주피터 노트북에 다 나와있어요.

loss가 줄어들다가 튄 것은, 계수를 움직여 봤는데 아차 싶은 것. 다시 원래대로 돌아옴
learning rate은 처음에 세게 줘야함 (local optima 빠지지 않기 위해서)
그래서 learning rate decay가 존재함.

loss도 hyperparameter.

loss가 달라지는것은
1. filter를 선언했을 때 계수들이 random. 그래서 시작점 부터 다르다.
2. 학습을 할 수록 (새로운 epoch 일수록), data를 shuffling을 한다. 순서를 똑같이 하면 순서를 외워버릴 수가 있잖아요. 그래서 새 epoch마다 data 순서를 바꿔요.


layer가 깊어질수록(feature map의 depth가 깊어질수록) resolution이 줄어들어요
이유1. 메모리가 줄어들어서


FC : 2차원이 1차원으로 바뀜 (max pooling하고 vector로 만듬) (공간정보 없어짐)



