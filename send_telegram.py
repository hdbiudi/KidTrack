import telegram

def send_telegram(photo_path="file_test/alert.png"):
    try:
        my_token = "5631050988:AAHE2y4t4TlcywVxjD392B1vKS0jpgdzFlU"
        chat_id = "5181950647"
        # Tạo bot
        bot = telegram.Bot(token=my_token)
        bot.sendPhoto(chat_id=chat_id, photo=open(photo_path, "rb"), caption='Nguy Hiểm')
    except Exception as ex:
        print("can not send message telegram", ex)
    print("send sucess")

# asyncio.run(send_telegram())
# tạo account telegram
# tạo Bot với BotFather
# /newbot
# đặt tên cho Bot
# lấy token
# Tạo group chat và lấy id
# https://api.telegram.org/bot[TOKEN]/getUdates
