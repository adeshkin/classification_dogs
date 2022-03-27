import logging
from aiogram import Bot, Dispatcher, types, executor
import os
import torch
from PIL import Image
from utils import load_model, prepare_img, ID2NAME
from config import TOKEN, WEBHOOK_URL, WEBHOOK_PATH, WEBAPP_HOST, WEBAPP_PORT


bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)

model = load_model()


def predict_label(img):
    img = prepare_img(img)
    with torch.no_grad():
        output = model(img)[0]
    probs = torch.softmax(output, dim=0)
    top3_labels = []
    top3_probs = []
    for label_id in torch.topk(probs, 3).indices.numpy():
        top3_labels.append(ID2NAME[label_id])
        top3_probs.append(probs[label_id].item())
    return top3_labels, top3_probs


@dp.message_handler(commands='start')
async def start(message: types.Message):
    await message.answer('Send photo')


@dp.message_handler(content_types=['photo'])
async def predict(message: types.Message):
    filename = 'img_' + str(message.from_user.id) + '.jpg'
    await message.photo[-1].download(destination_file=filename)
    img = Image.open(filename)
    os.remove(filename)
    top3_labels, top3_probs = predict_label(img)
    result = ''
    for i, (label, prob) in enumerate(zip(top3_labels, top3_probs)):
        if i != 0 and prob < 0.02:
            break
        prob = int(prob * 100)
        result += f'*{label}* : {prob}%\n'

    await bot.send_message(chat_id=message.from_user.id, text=result, parse_mode='Markdown')


async def on_startup(dp):
    logging.warning('Starting connection. ')
    await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)


async def on_shutdown(dp):
    logging.warning('Bye! Shutting down webhook connection')


if __name__ == '__main__':
    executor.start_webhook(
         dispatcher=dp,
         webhook_path=WEBHOOK_PATH,
         on_startup=on_startup,
         on_shutdown=on_shutdown,
         skip_updates=True,
         host=WEBAPP_HOST,
         port=WEBAPP_PORT)
