from tkinter import *
from PIL import ImageTk, Image
import os
import subprocess
import recognise
import training 
import get_face

def recon():
    recognise.recognise_face()
def train():
    training.training_img()
def get_img():
    get_face.get_face()

class Main:
    def __init__(self):
        self.root = Tk()
        self.root.title("Face recognition")
        self.root.geometry('1000x500')
        self.img = ImageTk.PhotoImage(Image.open('background.jpg'))
        print(self.img)
        self.back_image = Label(self.root,image=self.img)
        self.back_image.place(x=0,y=0,relwidth=1,relheight=1)

    def initialize(self):
        button1 = Button(self.root,text='Get Face',bg='cyan',fg='black',bd=0,font=('times new roman',15),command=get_img)
        button1.place(x=50,y=100,width=500,height=40)
        button2 = Button(self.root,text='Train Model',bg='cyan',fg='black',bd=0,font=('times new roman',15),command=train)
        button2.place(x=50,y=170,width=500,height=40)
        button3 = Button(self.root,text='Recognise face',bg='cyan',fg='black',bd=0,font=('times new roman',15),command=recon)
        button3.place(x=50,y=235,width=500,height=40)
        self.root.mainloop()
    def login(self):
        entry = Entry(self.root,bg="")
        button1 = Button(self.root,text='Recognise face',bg='cyan',fg='black',bd=0,font=('times new roman',15))
        button1.place(x=50,y=235,width=500,height=40)
gui = Main()
gui.initialize()
