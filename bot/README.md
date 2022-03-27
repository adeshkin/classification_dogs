### Dog breed classification
* [Telegram Bot](https://t.me/class_dog_bot)
* [Heroku](https://www.heroku.com/)

**Instruction:**
```bash
git clone https://github.com/adeshkin/classification_dogs
cp checkpoint/resnet18_best.pth bot
cd bot
heroku container:push --app class-dog web
heroku container:release --app class-dog web
heroku logs --tail --app class-dog  
```

