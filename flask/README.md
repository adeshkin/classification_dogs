### Dog breed classification
* [Localhost](http://192.168.0.100:5001/)

**Instruction:**
```bash
git clone https://github.com/adeshkin/classification_dogs
cp checkpoint/resnet18_best.pth flask
cd flask
docker build . -t class-dog 
docker run --name class-dog-app -p 5001:5001 class-dog
```

