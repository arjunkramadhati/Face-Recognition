import requests

import time

class message():
    def __init__(self):
        #self.bot_token = '655605995:AAFByuXQ7ZL2RAfw8-N_IP9HjxESt6NWxBU'
        self.bot_token = '782461724:AAGYEavQeNcfWbGOH5_nvXAGupGEAV9Eblk'
        #self.bot_chatID = '-245155974'
        self.bot_chatID = '-340772720'
        self.operatorID = '922474722'
        
    def telegram_bot_sendtext(self,bot_message,operator_message):
        self.send_text = 'https://api.telegram.org/bot' + self.bot_token + '/sendMessage?chat_id=' + self.bot_chatID + '&parse_mode=Markdown&text=' + bot_message
        self.send_text2 = 'https://api.telegram.org/bot' + self.bot_token + '/sendMessage?chat_id=' + self.operatorID + '&parse_mode=Markdown&text=' + operator_message
        requests.get(self.send_text)
        requests.get(self.send_text2)

    def report(self, bot_message,operator_message):
        self.telegram_bot_sendtext(bot_message,operator_message)

    def init_scheduler(self, bot_message,operator_message):
        schedule.every(2).minutes.do(self.report,bot_message,operator_message)

    def check_scheduler(self):
        schedule.run_pending()
  

#while True:
    #schedule.run_pending()
    #time.sleep(1)

#self.telegram_bot_sendtext("INFO: Operator missing","Report back to the machine")

tester = message()

#####Case 1: Operator goes missing and comes back after alert ##########
#tester.report("[ALERT] OPERATOR IDxxxx Missing from Machine 1 from TIMESTAMP","[CENTRAL ALERT] PLEASE REPORT BACK TO YOUR BOOTH")

#####Case 2: Operator goes missing and does not come back after 2 minutes ###########
tester.report("[ALERT] OPERATOR IDxxxx Missing from Machine 1 from TIMESTAMP","[CENTRAL ALERT] PLEASE REPORT BACK TO YOUR BOOTH")
##operator has not shown up 2 mins after alert#####
'''tester.init_scheduler("[ALERT] OPERATOR IDxxxx Missing UPDATE: Still Missing","[CENTRAL ALERT] PLEASE REPORT BACK TO YOUR BOOTH")
while True:
    tester.check_scheduler()
    time.sleep(1)'''


