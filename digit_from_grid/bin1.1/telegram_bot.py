import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler, ConversationHandler, updater
from telegram import InlineKeyboardButton, InlineKeyboardMarkup,  ReplyKeyboardRemove, KeyboardButton, ReplyKeyboardMarkup, update
from config import TOKEN
from sudoku_solver import sudoku_solver
from model_class import get_Prediction
from datetime import datetime
import os
import logging

class TelegramBot():
    def __init__(self):
        self.bot = telegram.Bot(token=TOKEN)
        self.updater = Updater(token = TOKEN)
        # ONLY FOR TESTING PORPUSES
        self.status = 0
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.RESTART ,self.MENU, self.PHOTO, self.CHECK_DIGIT, self.TESTING = range(5)

    def start(self,  update, context):
        
        if update.message is not None: 
            self.username = str(update.message.from_user.username)
            self.user_id = update.message.from_user.id
            self.chat_id = update.message.chat_id
        #print(self.chat_id)
        outgoing_message_text = "Hi {}\nWelcome to Sudoku Solver Bot 🔢, a computer vision project. 💻📹".format(self.username)
        gif_link = "https://media.giphy.com/media/l2JBygxaUuh8aJ6YHn/giphy.gif"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text='Solve a Sudoku 🙇‍♀️', callback_data='solve')],
            [InlineKeyboardButton(
                text='Learn more about the project 🧐', callback_data='learn')],
            [InlineKeyboardButton(
                text='Exit the bot 👋', callback_data='exit')]
        ])
        
        self.bot.sendAnimation(chat_id = self.chat_id, animation = gif_link, caption = outgoing_message_text, reply_markup = keyboard)
        #Return the callback handler with the new state
        return self.MENU
    
    def on_callback_query_menu(self, update, context):
        self.chat_id = update.effective_user.id
        query_data = update.callback_query.data
        print(query_data)
        if query_data == 'solve':
            outgoing_message_text = "Great 💪! Send us a photo 📸 of sudoku to solve!"
            self.bot.sendMessage(chat_id=self.chat_id,
                                 text=outgoing_message_text)
            #Return the callback handler with the new state
            return self.PHOTO

        elif query_data == 'learn':

            repo_link = "https://github.com/Tobias-GH-Schulz/digit_dataset"
            outgoing_message_text = "This is our github repo 📝, have a look and let us know for any issue 📪 or what do you think 🗣. \n{}".format(
                repo_link)

            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    #SAME LINE BUTTON
                    InlineKeyboardButton(text='Go Back to the Menu 🔁',
                                         callback_data='restart'),
                ]
            ])
            self.bot.sendMessage(chat_id=self.chat_id,
                                 text=outgoing_message_text, reply_markup=keyboard)
            return self.RESTART
        elif query_data == 'exit':
            self.endFall(update, context)

    def get_sudoku(self, update, context):
        
        self.save_path_temp = "digit_from_grid/bin1.1/temp_file/temp_file.jpg"
        
        self.file_image = self.bot.get_file(update.message.photo[-1].file_id)
        self.file_image.download(self.save_path_temp)
        
        predictor = get_Prediction(self.save_path_temp)
        self.matrix = predictor.get_matrix()
       
        outgoing_message_text = str(
            self.matrix) + "\n\nAre all the digit scanned correct 🧐?"
        #CHECK IF THE DIGIT DETECTED ON THE IMAGE ARE CORRECT
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
            #SAME LINE BUTTON
            InlineKeyboardButton(text='Yes ✅',
                                  callback_data='correct'),
            InlineKeyboardButton(
                text='No ❌', callback_data='not_correct')
            ]
        ])
        
        self.bot.sendMessage(chat_id=self.chat_id,
                             text=outgoing_message_text, reply_markup = keyboard)

        return self.CHECK_DIGIT
    
    def check_digit(self, update, context):
        query_data = update.callback_query.data
        if query_data == 'correct':

            solver = sudoku_solver(self.matrix)
            self.matrix_solved = solver.get_solved_sudoku()

            outgoing_message_text = str(
                self.matrix_solved) + "\n\nHere is the solution of your sudoku"
        #CHECK IF THE DIGIT DETECTED ON THE IMAGE ARE CORRECT
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    #SAME LINE BUTTON
                    InlineKeyboardButton(text='Go Back to the Menu 🔁',
                                         callback_data='restart'),
                ]
            ])
            os.remove(self.save_path_temp)
            self.bot.sendMessage(chat_id=self.chat_id,
                             text=outgoing_message_text, reply_markup=keyboard)
            self.status = 0
            return self.RESTART

        elif query_data == 'not_correct':
            # Try again
            
            if self.status == 0:
                outgoing_message_text = 'Sorry about that 😓, please try again with a better image,\nSend a new image. 📸'
                self.bot.sendMessage(chat_id=self.chat_id,
                                    text=outgoing_message_text)
                self.status += 1
                return self.PHOTO
            
            elif self.status == 1:
                outgoing_message_text = "It doesn't seems to work 😰 \n Can we store your image for testing porpuses 📝?"
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [
                        #SAME LINE BUTTON
                        InlineKeyboardButton(text='Yes ✅',
                                             callback_data='store'),
                        InlineKeyboardButton(
                            text='No ❌', callback_data='not_store')
                    ]
                ])
                self.bot.sendMessage(chat_id=self.chat_id,
                                    text=outgoing_message_text, reply_markup = keyboard)
                self.status = 0
                return self.TESTING
                
        pass

    def check_to_store(self, update, context):
        query_data = update.callback_query.data
        
        if query_data == 'store':
            
            now = datetime.now()
            save_path = 'digit_from_grid/bin1.1/not_working_img/{}.jpg'.format(now.strftime("%d%m%Y_%H%M"))
            self.file_image.download(save_path)

            outgoing_message_text = "Thank you very much for your contribution. 🙏"
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    #SAME LINE BUTTON
                    InlineKeyboardButton(text='Go Back to the Menu 🔁',
                                         callback_data='restart'),
                ]
            ])
            self.bot.sendMessage(chat_id=self.chat_id,
                                 text=outgoing_message_text, reply_markup=keyboard)
            return self.RESTART
        
        elif query_data == 'not_store':
            
            os.remove(self.save_path_temp)
            outgoing_message_text = "Thank you for your time.\nAll the files uploaded are deleted. 🙏"
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    #SAME LINE BUTTON
                    InlineKeyboardButton(text='Go Back to the Menu 🔁',
                                         callback_data='restart'),
                ]
            ])
            self.bot.sendMessage(chat_id=self.chat_id,
                                 text=outgoing_message_text, reply_markup=keyboard)
            return self.RESTART
        

    def endFall(self, update, context):
        user_data = context.user_data
        if 'choice' in user_data:
            del user_data['choice']

        outgoing_message_text = "Thank you very much {} 🙌😄 \n To restart 🔄 just press or type /start".format(
            self.username)
        self.bot.sendMessage(chat_id = self.chat_id, text = outgoing_message_text, reply_markup = ReplyKeyboardRemove())

        user_data.clear()
        return ConversationHandler.END
        

    def execute(self):
        
        dp = self.updater.dispatcher

        conv_handler = ConversationHandler(
            #FIRST COMMAND SEND TO BOT
            entry_points=[CommandHandler('start', self.start)],
            conversation_timeout=60,
            allow_reentry=True,
            states = {
                self.RESTART: [CallbackQueryHandler(self.start)], 
                self.MENU: [CallbackQueryHandler(self.on_callback_query_menu)],
                self.PHOTO: [MessageHandler(Filters.photo, self.get_sudoku)],
                self.CHECK_DIGIT: [CallbackQueryHandler(self.check_digit)],
                self.TESTING: [CallbackQueryHandler(self.check_to_store)]
            },
           fallbacks=[MessageHandler(Filters.regex('^Done$'), self.endFall)] 
        ) 

        dp.add_handler(conv_handler)
        self.updater.start_polling()
        self.updater.idle()


if __name__ == '__main__':
    bot = TelegramBot()
    bot.execute()
