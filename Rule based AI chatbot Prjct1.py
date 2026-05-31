'''Decode Labs Artifical Intelligence Project 1 
SIMPLE RULE BASED AI CHATBOT-
LENU-a friendly chatbot
done by:SHENZA P M'''
response={"hello":"Hello there!I'm lenu,what can i help u with today?",
          "hi":"Hi dear,feel free to ask me anything ,Im glad to help.",
          "hey":"Hey! What's up my friend,What are you up to?",
          "bye":"bye-bye,Dont forget me.Always happy to help!",
          "goodbye":"Goodbye friend,Have a nice day!",
          "how are you?": "I'm fine,What about you?",
          "sad":"I'm sorry you're having a tough day. Maybe take a short walk and come back refreshed.",
          "happy":"That's awesome! Keep the positive energy going.",
          "who are you?":"I'm lenu,a friendly chatbot.How are you doing?",
          "lenu":"That's me :) ,your friend",
          "what all can you do for me?":"I can help you out on things you are stuck on and be your friend when in need!",
          "what is a chatbot?":"A chatbot is a program that simulates conversation — like me!",
          "what is decodelabs":"DecodeLabs is the company running your AI internship. Keep building!",
          "what is project 1" :"Project 1 is the Rule-Based AI Chatbot!",
          "study tips":"Study for 25 minutes and take a 5 minute break.",
          "motivate me":"Small progress every day becomes huge success.",
          "help":"Available commands → Greetings:hello,hi,hey|Mood:happy,sad|Info:who are you?,what is a chatbot?,what is decodelabs?,what is project 1?,suggest a movie|Support:study tips,motivate me|Other:lenu,bye, exit",
          "suggest a movie":"oo there's soo manyy...,I would suggest 'Dead Poets Society' ",
 
          }
fallback = "I'm still learning!Type 'help' to explore everything I can do."
#helper functions
def sanitize(raw:str)->str:
    return raw.lower().strip()
 
 
def get_response(clean_input:str)->str:
    return response.get(clean_input,fallback)
 
 #main loop
def run_chatbot():
    print("Lenu — Friendly DecodeLabs AI Chatbot")
    print("Type 'exit' to quit.")
 
    while True:                           
        raw_input=input("\nYou: ")
        clean_input=sanitize(raw_input)
 
        if clean_input=="exit":         
            print("Lenu: Goodbye! Session ended.")
            break
 
        reply=get_response(clean_input)
        print(f"Lenu: {reply}")
#entry point
if __name__ == "__main__":
    run_chatbot()

'''Main takeaway from Project 1:
1. Control flow - a while loop keeps the chatbot alive until told to stop.
2. Dictionary   - O(1) instant lookup beats a slow if-elif chain every time.
3. Sanitization - clean the input first so matching always works.
4. IPO model    - every program is just Input, Process, and Output.
5. White box AI - rule-based means zero surprises, fully traceable output.'''
