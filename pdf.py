from fpdf import FPDF
from datetime import datetime

"""
Copyright 2013 by SVM Risk Consulting
All rights reserved. No part of this document may be reproduced or transmitted in any form or by any means, electronic, mechanical, photocopying, recording, or otherwise, without prior written permission of SVM Risk Consulting.
"""

title='Lending Club Loan Applicant Ranking - '+datetime.now().date().strftime('%Y-%m-%d')

def isodd(num):
    return num & 1 and True or False

class PDF(FPDF):
        def header(self):
                #Arial bold 15
                self.set_font('Arial','B',15)
                #Calculate width of title and position
                w=self.get_string_width(title)+6
                self.set_x((210-w)/2)
                #Colors of frame, background and text
                self.set_draw_color(0,80,180)
                self.set_fill_color(230,230,0)
                self.set_text_color(220,50,50)
                #Thickness of frame (1 mm)
                self.set_line_width(1)
                #Title
                self.cell(w,9,title,1,1,'C',1)
                #Line break
                self.ln(10)

        def footer(self):
                #Position at 1.5 cm from bottom
                self.set_y(-15)
                #Arial italic 8
                self.set_font('Arial','I',8)
                #Text color in gray
                self.set_text_color(128)
                #Page number
                self.cell(0,10,'Page '+str(self.page_no()),0,0,'C')

        def chapter_title(self,num,label):
                #Arial 12
                self.set_font('Arial','',12)
                #Background color
                self.set_fill_color(200,220,255)
                #Title
                self.cell(0,6,"%s"%(label),0,1,'L',1)
                #Line break
                self.ln(4)

        def chapter_body(self,name, offset):
                #Read text file
                txt=file(name).read()
                #Times 12
                self.set_font('Times','',8)
                #Output justified text
                self.set_fill_color(256,256,256)
                self.cell(15, 5 , 'Ranking',
                          border=1, fill=1,)
                self.cell(30, 5 , 'Probability of default %',
                          border=1, fill=1,)
                self.cell(15, 5 , 'Offer id',
                          border=1, fill=1,)
                self.cell(20, 5 , 'Loan amount',
                          border=1, fill=1,)
                self.cell(20, 5 , 'Funded amount',
                          border=1, fill=1,)
                self.cell(10, 5 , 'Term',
                          border=1, fill=1,)
                self.cell(12, 5 , 'APR %',
                          border=1, fill=1,)
                self.cell(40, 5 , 'Purpose',
                          border=1, fill=1,)
                self.cell(30, 5 , 'Status',
                          border=1, fill=1,)
                self.ln()
                self.cell(15, 5 , '[SVM Cons]',
                          border=1, fill=1,)
                self.cell(30, 5 , '[SVM Consulting]',
                          border=1, fill=1,)
                self.cell(15, 5 , '[LendingC]',
                          border=1, fill=1,)
                self.cell(20, 5 , '[LendingClub]',
                          border=1, fill=1,)
                self.cell(20, 5 , '[LendingClub]',
                          border=1, fill=1,)
                self.cell(10, 5 , '[LC]',
                          border=1, fill=1,)
                self.cell(12, 5 , '[LC]',
                          border=1, fill=1,)
                self.cell(40, 5 , '[LendingClub]',
                          border=1, fill=1,)
                self.cell(30, 5 , '[LendingClub]',
                          border=1, fill=1,)
                self.ln()
                for i,t in enumerate(txt.split("\n")):
                    try:
                        #default, status, offid, url,amnt,funded_amnt, term,  apr, purpose, lat, lon = t.split(",")
                        default, status, offid, url, amnt, \
                            funded_amnt, term,  apr, purpose = t.split(",")
                        d_f = float(default.replace("[",""))*100
                        if d_f < 30.:
                            self.set_fill_color(127,255,0)
                        else:
                            self.set_fill_color(255,0,0)
                        self.cell(15, 5 , str(i+offset),
                                  border=1, fill=1,)

                        #if isodd(i):
                        #    self.set_fill_color(256,256,256)
                        #else:
                        #    self.set_fill_color(230,230,230)
                        self.cell(30, 5 , str(d_f)[0:5]+'%',
                                  border=1, fill=1,)
                        self.cell(15, 5 , offid,
                                  border=1, fill=1,link=url)
                        self.cell(20, 5 , amnt,
                                  border=1, fill=1,)
                        self.cell(20, 5 , funded_amnt,
                                  border=1, fill=1,)
                        self.cell(10, 5 , term,
                                  border=1, fill=1,)
                        self.cell(12, 5 , apr[0:5]+'%',
                                  border=1, fill=1,)
                        self.cell(40, 5 , purpose.replace("'",""),
                                  border=1, fill=1,)
                        self.cell(30, 5 , status,
                                  border=1, fill=1,)
                        #Line break
                        self.ln()
                    except:
                        print 'failed: '+str(i)
                #Mention in italics
                self.set_font('','I')
                self.cell(0,5,'(end of excerpt)')

        def print_chapter(self,num,title,name):
                self.add_page()
                self.chapter_title(num,title)
                self.chapter_body(name, num)
