# cuda_nn
Most of these sources were developed in cuda language for deep learning.

And since it is not yet complete, we plan to continue developing it.

My environment is:

1. windows 10
2. visual studio 2017
3. cuda v11.8
4. nvidia geforece gtx1080ti 11G
5. intel i7-12700
6. ram 16G

Thank you!!

================= jindallae =================

model class description:

모델 생성자는 링크 클래스와 실행 레이어 클래스의 다중 상속 되어 있습니다.

훈련과 추측 실행시 실제 사용자가 접근할 수 있는 클래스 이기도 하여 중심이 되는 클래스 입니다.

생성자 함수에서 사용자가 정의한 레이어의 인풋과 아웃풋을 인자로 입력 받으면 

새로운 자식 링크객체가 생성되고 부모의 실행 레이어 객체는 포인터로 가리키게 하게 했습니다.

![model_01](https://user-images.githubusercontent.com/36714695/209514656-637696a8-416f-4c3e-aaf7-5f2dc610d9fc.jpg)

또한 새로운 자식 링크끼리 연결은 부모의 링크를 토대로 아웃풋 노드에서 출발해 인풋노드 까지 부모링크 객체를 마킹을 합니다. (is_selected)

그런 다음 다시 인풋에서 시작하여 아웃풋까지 마킹 자국을 따라 가면서 임시로 마킹된 것만 저장 합니다.

이렇게 선택 되어진 링크들을 가지고 자식 노드를 생성 및 연결을 해줍니다.

![model_02](https://user-images.githubusercontent.com/36714695/209516947-e261af02-a6e7-47bc-857b-fe2b110688d8.jpg)

