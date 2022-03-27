import logging
from aiogram import Bot, Dispatcher, types, executor
from utils import predict_label

TOKEN = ''
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)


@dp.message_handler(commands='start')
async def start(message: types.Message):
    await message.answer('Send photo')


@dp.message_handler(content_types=['photo'])
async def predict(message: types.Message):
    img = None
    top3_labels, top3_probs = predict_label(img)
    result = ''
    for label, prob in zip(top3_labels, top3_probs):
        result += f'*{label}*: {prob}\n'
    await message.answer(result, parse_mode='MarkdownV2')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)