## WEEK1

> RNN : 말이 긍정이다/부정이다 표현 할 수 있음

> GAN : input으로 noise를 준다.

* 1주차 segmentation    
* 2주차 word generation (예상하는 모델 학습)    
*('나는'이라고 입력주면 '지금'을 뱉고, '지금'을 넣으면 '랩실'을 뱉고, '랩실'을 넣으면 '있다'를 뱉고. '있다'를 넣으면 '끝'이라는 포인트)*     
* 3주차
이미지로 부터 세그멘테이션 했다 -> 특징을 추출햇다     
워드 제너레이션 -> 워드로 부터 특징을 추출함.    
2개를 결합해서, 영상을 주고 영상에 대한 이미지 캡셔닝을 함.    
이미지 캡셔닝이 되면 비디오 캡셔닝이 되죠.    

> Segmentation

panoptic segmentation은 instance segmentation에서 하나 더 나아가서, 배경까지 segmentation 하는 것    

> batch 와 epoch

batch(=iter) & update(학습) ∈ epoch(shuffle)    

---

실습 (공부 후 수정 필요)

FCN
FC 대신 DeConv가 있어서 뻥튀기가 되었다.
(공간정보를 살려서 가다가 upsampling을 하자.

fixme를 하나씩 수정해주세요. 
stride 2는 2x2를 의미해요

deconv로 256,256으로 올라갑니다.

opendata set이라고 medical dataset을 구할 수 있어요.


https://research.google/pubs/pub45732/
논문저널.


--- 

## WEEK2


> 지난주 질문
    
DICOM xray
-> DICOM 최근 스펙문서에 의하면 default type으로 JPEG2000이 지정되어 있습니다. JPEG도 다양한 버전이 있는데, JPEG2000의 경우 lossless입니다. 제가 수업시간에 언급한 JPEG는 카메라나 저희가 컴퓨터비전에서 image file을 save할 때 보통 사용되는 일반JPEG로써 2000과는 다릅니다.     
그러니, 원본이 복원되는 lossless JPEG인 경우일 확률이 크며, 걱정하지 않으셔도 됩니다.    
Sante DICOM viewer에서 제공하는 DICOM Hexa viewer 기능을 통해 제가 갖고있는 CT 데이터로 한번 살펴봤는데, 그림의 아래처럼 lossless JPEG이라고 나와있는 것을 확인할 수 있었습니다.    
DICOM header: https://www.dicomlibrary.com/dicom/transfer-syntax/    
DICOM viewer: https://www.santesoft.com/win/sante-dicom-viewer-pro/sante-dicom-viewer-pro.html (edited)     

> 지난주 사전질문
1. MLP로 쌓아도 되는데 왜 Conv 를 쓰는가? => 유의한 feature를 뽑기위해(2차원정보)

> 1주차 리뷰


> Word Generator

