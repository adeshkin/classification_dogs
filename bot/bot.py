import os
import logging
from aiogram import Bot, Dispatcher, types, executor
import torch

from scripts.model import load_model


TOKEN = ''
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)

model = load_model()


@dp.message_handler(commands='start')
async def start(message: types.Message):
    await message.answer('Отправьте фото')


@dp.message_handler(content_types=['photo'])
async def predict(message: types.Message):
    download_dir = 'uploads/'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    img_name = 'img' + str(message.from_user.id) + '.jpg'
    await message.photo[-1].download('./img/' + img_name)
    # predict
    img = prepare_img('./uploads/' + img_name)
    output = model(img)
    _, pred_label = torch.max(output, 1)

    os.remove('./uploads/' + img_name)

    await message.answer(f'Predicted label = {pred_label.numpy()[0]}')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)