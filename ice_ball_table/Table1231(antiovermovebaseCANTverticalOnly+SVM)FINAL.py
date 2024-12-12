# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:35:19 2020

@author: KYC-201908
"""


import cv2
import numpy as np
import cv2 as cv
import serial 
import math

import time
import joblib

UARTENABLE=0#測試控制項####測試控制項

SVMENABLE=1#SVM控制項####SVM控制項


COM_PORT = 'COM4'#+str(COM_PORT)  # 請自行修改序列埠名稱
BAUD_RATES = 115200
if(UARTENABLE==1): 
   ser = serial.Serial(COM_PORT, BAUD_RATES)


#----------------
sq1 = 195 
sq2 = 600
sq3 = 220
sq4 = 1040
#影像裁切四角

ballhalf=0 #球中心偏移

endline = 55; #滑軌位置/預測線

hitendline = 100 #打擊線

midline = 400 #中線/復歸線

attline = 305 #打擊線


upline = 85 #上邊界座標
downline =415 #下邊界座標


#----------------
ballxreal=10000
ballyreal=10000

ballxrealp=100000
ballyrealp=100000


ballK=0
ballKp=0

edgex=10000
edgey=10000


handxreal=10000
handyreal=10000

handxrealp1=100000
handyrealp1=100000

handxrealp2=100000
handyrealp2=100000


ballxA=-23
ballxB=5
ballyA=-22
ballyB=3
#數值歸零

roundn=0 #幀數閃爍燈號

attlock=0 #打擊發送鎖

SVMlock=0

#----------------
p4=0
bp4=0
p7=0
p5=0
p6=0

spd10f=0
spd10fp1=0
spd10fp2=0
spd10fp3=0
spd10fp4=0
spd10fp5=0
spd10fp6=0
spd10fp7=0
spd10fp8=0
spd10fp9=0
spd10fp10=0


#球速計算歸零



fourcc = cv2.VideoWriter_fourcc(*'XVID')


#out = cv2.VideoWriter('output.avi', fourcc, 60, (640, 480))



prinw=0



#----------------
if(UARTENABLE==1):
  cap = cv2.VideoCapture(0)
  
  #cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
else:
  #cap = cv2.VideoCapture("2020-09-15_21-18-12.mp4")

  #cap = cv2.VideoCapture("2020-09-14_21-03-45.mp4")

  cap = cv2.VideoCapture("C:\PYIB\DATA\WIN_20211125_21_23_40_Pro.mp4")
#影像輸入


#----------------
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
#影像輸入解析度


# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# cap = cv2.VideoCapture()
# cap.open(0 + cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FOURCC, fourcc)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FPS, 60)


fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)


#cap = cv2.VideoCapture("2021-05-10_05-07-16.mp4")


for i in range(1):
    ret, framefirst = cap.read()#第一帧儲存/相減


#framefirst = framefirst[sq1:sq2,sq3:sq4]
#framefirst = cv2.resize(framefirst, (1040, 600), interpolation=cv2.INTER_NEAREST)


kernel = np.ones((5,5), np.uint8)#擴張侵蝕核


#----------------
# define range of color in HSV 
lowerR1 = np.array([80/2,20*2.55,30*2.55])
upperR1 = np.array([150/2,100*2.55,100*2.55])
#球色彩辨識/綠
lowerY1 = np.array([40/2,55*2.55,25*2.55])
upperY1 = np.array([60/2,100*2.55,90*2.55])
#手把色彩辨識/黃


# lowerR1 = np.array([90/2,40*2.55,10*2.55])
# upperR1 = np.array([160/2,100*2.55,60*2.55])

framefirst = cv2.GaussianBlur(framefirst, (13, 13), 0)#第一帧儲存/相減 高斯模糊

hsv1 = cv2.cvtColor(framefirst, cv2.COLOR_BGR2HSV)#第一帧儲存/相減 BGR-HSV轉換
dframe = cv2.inRange(hsv1, lowerR1, upperR1)#第一帧儲存/相減 綠色濾鏡
  


endlinerulera=upline#滑軌位置/預測線畫線起始點
endlinerulerb=22#滑軌位置/預測線畫線間隔



URpositionP3=4
URpositionP2=4
URpositionP1=4
URposition=4#UART發送歸零


SVMmodel = joblib.load("svclassifier.pkl")
I=0

SVMAns=0
if cap.isOpened():
    while(cap.isOpened()):
       start = time.perf_counter()
       
       ret, framenow = cap.read()#攝影帧讀取
       framenow = cv2.GaussianBlur(framenow, (13, 13), 0)     #攝影帧 高斯模糊

#------------------------------------------------------------------
    
       hsv2 = cv2.cvtColor(framenow, cv2.COLOR_BGR2HSV)#攝影帧 BGR-HSV轉換
       ddframe = cv2.inRange(hsv2, lowerY1, upperY1)#攝影帧 黃色濾鏡

    
       
       absd = cv2.absdiff(dframe, ddframe) #攝影帧與第一帧相減
       
            
       
       erosion = cv2.erode(absd, kernel, iterations = 1)#侵蝕
       dilation = cv2.dilate(erosion, kernel, iterations = 1)#擴張
       
       
       contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#尋找輪廓球心
       center_x=10000
       center_y=10000 
       for cnt in range(len(contours)):
           duobufx=10000
           duobufy=10000

           # 提取與繪制輪廓
           cv.drawContours(framenow, contours, cnt, (0, 255, 255), 1)
           
           
           if len(contours)>1:
               duobufx=center_x
               duobufy=center_y
           
           M = cv2.moments(contours[cnt])  # 計算第一條輪廓的各階矩,字典形式
           #center_x = int(M["m10"] / M["m00"])
           center_y = int(M["m01"] / M["m00"])
                
           x,y,w,h = cv.boundingRect(contours[cnt])         
           center_x =x
           
           if len(contours)>1:
               if duobufx<center_x:
                   center_x=duobufx
                   center_y=duobufy       


           if cnt==len(contours)-1:
               cv2.circle(framenow, (center_x, center_y), 1, (0, 255, 255), -1)#繪製中心點       
               handxreal = int(np.mean(center_x))
               handyreal = int(np.mean(center_y))            
  
#------------------------------------------------------------------          

       #framenow = framenow[sq1:sq2,sq3:sq4]
       #framenow = cv2.resize(framenow, (1040, 600), interpolation=cv2.INTER_NEAREST)
       
       
       hsv2 = cv2.cvtColor(framenow, cv2.COLOR_BGR2HSV)#攝影帧 BGR-HSV轉換
       ddframe = cv2.inRange(hsv2, lowerR1, upperR1)#攝影帧 綠色濾鏡

    
       
       absd = cv2.absdiff(dframe, ddframe) #攝影帧與第一帧相減
       
            
       
       erosion = cv2.erode(absd, kernel, iterations = 2)#侵蝕
       dilation = cv2.dilate(erosion, kernel, iterations = 2)#擴張
       
       
       contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#尋找輪廓球心
       for cnt in range(len(contours)):

           # 提取與繪制輪廓
           cv.drawContours(framenow, contours, cnt, (0, 255, 0), 1)
           
           M = cv2.moments(contours[cnt])  # 計算第一條輪廓的各階矩,字典形式
           center_x = int(M["m10"] / M["m00"])
           center_y = int(M["m01"] / M["m00"])
           
           ballxreal = int(np.mean(center_x))
           ballyreal = int(np.mean(center_y))
           cv2.circle(framenow, (center_x, center_y), 1, (0, 255, 0), -1)#繪製中心點
           #cv2.circle(framenow, (center_x-ballhalf, center_y), 1, (0, 255, 0), -1)#繪製中心點
           
           #print(np.mean(center_x),np.mean(center_y),sep=",")          

#------------------------------------------------------------------              
       if(SVMENABLE==1):
           if (handxreal < handxrealp1)and(handxrealp1-handxreal>5)and(downline>handyreal>upline)and(endline<ballxreal)and(SVMlock==0) : 
    
    
               if(downline>ballyreal>upline):
                   b3=np.array([handxreal,handyreal])-np.array([ballxreal,ballyreal])
                   ballhand =math.hypot(b3[0],b3[1])   
                   if(ballhand<45):
                       predOut = SVMmodel.predict([[handxreal,handyreal,handxrealp1,handyrealp1,ballxreal,ballyreal]])
                       if(predOut==1):
                                SVMAns=1
                                if(UARTENABLE==1):
                                  ser.write(str(14140).encode())                     
                                  print('4')
                                  
                       elif(predOut==2):
                                SVMAns=3
                                if(UARTENABLE==1):
                                  ser.write(str(20200).encode())                     
                                  print('10')
                                  
                           
                       I=I+1
                       prinw=predOut
                       #print('SVM'+str(predOut))
                       SVMlock=1             

               
           if(ballxreal<endline*3):
               SVMlock=0
               
           
#---------------------------------------------------------

       #print(ballxrealp-ballxreal) 
       
       #----------
       # p4=0
       # bp4=0
       
       p1=np.array([ballxreal,ballyreal])
       p2=np.array([ballxrealp,ballyrealp])
       p3=p2-p1
       p5=math.hypot(p3[0],p3[1])
           
       p4= ballxrealp-ballxreal
     
       attline=int(p4)*5+endline
       #球速計算

       if (ballxreal < ballxrealp)and(p4>5)and(downline>ballyreal>upline)and(endline<ballxreal) :#反過動條件
            
            #----------
            c=np.array([[ballxreal,1],[ballxrealp,1]])
            d=np.array([ballyreal,ballyrealp])
            ans=np.linalg.solve(c,d)
            #print(ans)#1,2
            a=ans[0]
            b=ans[1]
           
            YY=a*endline+b
            ZZ=a*hitendline+b
            #第一段直線預測
            
            ballK=(ballyrealp-ballyreal)/(ballxrealp-ballxreal)
            

            if  abs(ballKp-ballK)<0.5:
                #cv2.line(framenow, (ballxreal,ballyreal), (endline, int(YY)), (0, 0, 255), 2)           
                if (YY>upline) and (YY<downline):                       
                    cv2.line(framenow, (ballxreal,ballyreal), (endline, int(YY)), (0, 0, 255), 2)#直預測線 畫線
                 
                    #----------
                    bp1=np.array([ballxreal,ballyreal])
                    bp2=np.array([endline,int(YY)])
                    bp3=bp2-bp1
                    bp4=math.hypot(bp3[0],bp3[1])
                    #預測長度累計
                
                else:
                    
                    #----------
                    if(YY<upline):         
                        edgex=(upline-b)/a
                        edgey=upline

                    elif(YY>downline):              
                        edgex=(downline-b)/a  
                        edgey=downline
                        
                    cv2.line(framenow, (ballxreal,ballyreal), (int(edgex), int(edgey)), (0, 0, 255), 2)#第一段反射 畫線
                    
                    #----------
                    bp1=np.array([ballxreal,ballyreal])
                    bp2=np.array([int(edgex),int(edgey)])
                    bp3=bp2-bp1
                    bp4=math.hypot(bp3[0],bp3[1])
                    #預測長度累計
    
                    if(a!=0):#防反射抖動/迴轉
                        
                    #----------     
                      if(ballyreal < ballyrealp):        
                          
                          edgex=(upline-b)/a
                          edgey=upline
                          
                      elif(ballyreal > ballyrealp): 
                          
                          edgex=(downline-b)/a  
                          edgey=downline
                         
                    edgex=edgex-ballhalf        
                    refx=ballxreal-((ballxreal-edgex)*2)
                    refy=ballyreal
                    
                    c=np.array([[edgex,1],[refx,1]])
                    d=np.array([edgey,refy])
                    ans2=np.linalg.solve(c,d)
                    #print(ans)#1,2
                    a=ans2[0]
                    b=ans2[1]
                 
                    YY=a*endline+b#第二段直線預測
                    ZZ=a*hitendline+b

                    if (YY>upline) and (YY<downline):                       
                        cv2.line(framenow, (int(edgex),int(edgey)), (endline, int(YY)), (0, 0, 255), 2) #第二段直預測線      
                    
                        #----------
                        bp1=np.array([int(edgex),int(edgey)])
                        bp2=np.array([endline,int(YY)])
                        bp3=bp2-bp1
                        bp4+=math.hypot(bp3[0],bp3[1])
                        #預測長度累計
              
            
                    else:
                        if(a!=0):
                            
                        #----------
                          if(ballyreal < ballyrealp):        

                              edgex2=(downline-b)/a
                              edgey2=downline
                              refy2=upline
      
                              
                          elif(ballyreal > ballyrealp):

                              
                              edgex2=(upline-b)/a 
                              edgey2=upline
                              refy2=downline
                        
                        cv2.line(framenow, (int(edgex),int(edgey)), (int(edgex2),int(edgey2)), (0, 0, 255), 2) #第二段反射 畫線          
                        
                        #----------
                        bp1=np.array([int(edgex),int(edgey)])
                        bp2=np.array([int(edgex2),int(edgey2)])
                        bp3=bp2-bp1
                        bp4+=math.hypot(bp3[0],bp3[1])
                        #預測長度累計
                              
                        #----------
                        edgex2=edgex2-ballhalf       
                        
                        refx2=edgex-((edgex-edgex2)*2)
                        
                        
                        g=np.array([[edgex2,1],[refx2,1]])
                        h=np.array([edgey2,refy2])
                        ans2=np.linalg.solve(g,h)
                        #print(ans)#1,2
                        e=ans2[0]
                        f=ans2[1]
                     
                        YY=e*endline+f#第三段直線預測
                        ZZ=e*hitendline+f
                        cv2.line(framenow, (int(edgex2),int(edgey2)), (endline, int(YY)), (0, 0, 255), 2) #第二段直預測線          
    
                        #----------
                        bp1=np.array([int(edgex2),int(edgey2)])
                        bp2=np.array([endline, int(YY)])
                        bp3=bp2-bp1
                        bp4+=math.hypot(bp3[0],bp3[1])
                        #預測長度累計
         

                # p6=p5
                # p5=p4
                # p7=(p6+p5)/2#球速延遲反過動
                
                if ballxreal<attline and attlock==0:   #移動-打擊指令               
                   #cv2.circle(framenow, (600, 10), 1, (255, 0, 0), 10)   #打擊指示燈號
                   
                   URpositionP3=URpositionP2 
                   URpositionP2=URpositionP1 
                   URpositionP1=URposition
                   URposition=14-int((YY-upline)/endlinerulerb)
                   
                   URpositionZZ=14-int((ZZ-upline)/endlinerulerb)
                                       
                   if(URposition>=0) and (URposition<15):    
                      if(UARTENABLE==1):
                         ser.write(str(((URposition+10)*1000+(URpositionZZ+10)*10)+1).encode())                    
                      #print(URposition)     
                      attlock=1     
                          
                if attlock==0: #非打擊移動指令
                   URpositionP3=URpositionP2 
                   URpositionP2=URpositionP1 
                   URpositionP1=URposition
                   URposition=14-int((YY-upline)/endlinerulerb)
                   if(URposition>=0) and (URposition<15):  
                       if(URpositionP1!=URposition):
                           if(UARTENABLE==1):
                               
                          
                             if(abs(URposition-URpositionP1)>1):

                                 ser.write(str(((URposition+10)*1000+(URposition+10)*10)+0).encode())                    
                
                             else:
                                 ser.write(str(((round((URposition+URpositionP1+URpositionP2)/3)+10)*1000)+((round((URposition+URpositionP1+URpositionP2)/3)+10)*10)+0).encode()) 








                               
                           #print(URposition)


            
            ballKp=ballK
      
            # if (abs(ballyrealp-ballyreal) > 1 ) :

            #     ballxrealp=ballxreal
            #     ballyrealp=ballyreal
       else:

             if(ballxreal>midline)and(ballxreal > ballxrealp):#滑軌復歸
                 URpositionP3=URpositionP2 
                 URpositionP2=URpositionP1 
                 URpositionP1=URposition
                 URposition=7
                 if(URposition>=0) and (URposition<15):
                    if(URpositionP1!=URposition):
                      if(UARTENABLE==1):
                        ser.write(str(((URposition+10)*1000+(URposition+10)*10)+0).encode())                     
                      #print(URposition)

                 attlock=0   
                 
             p5=0
             p6=0
             p7=0#速度計算變數歸零
            
       ballxrealp=ballxreal
       ballyrealp=ballyreal

       handxrealp2=handxrealp1
       handyrealp2=handyrealp1

       handxrealp1=handxreal
       handyrealp1=handyreal

       # spd10fp10=spd10fp9
       # spd10fp9=spd10fp8
       # spd10fp8=spd10fp7
       # spd10fp7=spd10fp6
       # spd10fp6=spd10fp5
       # spd10fp5=spd10fp4
       # spd10fp4=spd10fp3
       # spd10fp3=spd10fp2
       # spd10fp2=spd10fp1
       # spd10fp1=p5
       # #前帧紀錄
       end = time.perf_counter() 
       print(end - start)  
       
       # spd10f=(spd10fp1+spd10fp2+spd10fp3+spd10fp4+spd10fp5+spd10fp6+spd10fp7+spd10fp8+spd10fp9+spd10fp10)/10
          
             
       if ballxreal<attline:                     
           cv2.circle(framenow, (600, 10), 1, (255, 0, 0), 10)          
 
            
           
       #-----------------      
       if roundn==0:    
           cv2.circle(framenow, (615, 10), 1, (0, 255, 255), 10)  
           roundn=1
       elif roundn==1:
           cv2.circle(framenow, (630, 10), 1, (0, 255, 255), 10)
           roundn=0
       #幀數閃爍燈號
        
        
       cv2.putText(framenow, 'xspd:'+str(p4), (500, 25), cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 255), 1, cv2.LINE_AA)
       cv2.putText(framenow, 'dst:'+str(bp4), (500, 40), cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 255), 1, cv2.LINE_AA)
       #速度距離文字顯示
       cv2.putText(framenow, str(prinw)+str(I), (200, 40), cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 255), 1, cv2.LINE_AA)
       cv2.putText(framenow, 'spd:'+str(p5), (500, 55), cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 255), 1, cv2.LINE_AA)
   
    
       cv2.line(framenow, (endline, 0), (endline, 755), (0, 0, 255), 2)
       cv2.line(framenow, (midline, 0), (midline, 755), (0, 0, 255), 2)
       cv2.line(framenow, (attline, 0), (attline, 755), (0, 0, 255), 2)
       #滑軌線/中線/攻擊線 畫線
       
       # cv2.line(framenow, (attline+40, 0), (attline+40, 555), (0, 0, 255), 2)
       # cv2.line(framenow, (attline-40, 0), (attline-40, 555), (0, 0, 255), 2)

       cv2.line(framenow, (0, upline), (5555, upline), (0, 0, 255), 2)
       cv2.line(framenow, (0, downline), (5555, downline), (0, 0, 255), 2)
       #上下邊界 畫線
       if SVMAns==1:
           cv2.circle(framenow, (15, 355), 1, (0, 255, 255), 20)
       elif SVMAns==2:    
           cv2.circle(framenow, (15, 255), 1, (0, 255, 255), 20) 
       elif SVMAns==3:
           cv2.circle(framenow, (15, 155), 1, (0, 255, 255), 20)

       
       for i in range(1,15,1):
           cv2.line(framenow, (endline+20, endlinerulera+endlinerulerb*i), (endline-20, endlinerulera+endlinerulerb*i), (0, 0, 255), 2)
           #滑軌打擊區間 畫線
        
       cv2.imshow('img1',framenow)
       
       cv2.imshow('img2',absd)
           
       cv2.imshow('img3',ddframe)
       
       #out.write(framenow)
       

         
       if(UARTENABLE==1):
           k = cv2.waitKey(1) & 0xff
           if k==27:
               break
       else:
           k = cv2.waitKey(13) & 0xff
           if k==27:
               break
 
        
cap.release()
#out.release()    

    
    
cv2.destroyAllWindows()   