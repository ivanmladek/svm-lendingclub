import smtplib, os
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email.Utils import COMMASPACE, formatdate
from email import Encoders
from datetime import datetime

message = """
Dear Subscriber, 

We are enclosing the latest list of Lending Club loans. As usual
the green loans are the least risky, yellow loans are mid-risk and
red loans are the most risky.

Yours Truly, 

SVM Risk Consulting
svmriskconsulting@gmail.com
T: (302) 597-0301

DISCLAIMER:
Investing in Lending Club loans is inherently very risky.
-------------------------------
Risks of Investing
Lending Club Notes are offered pursuant to a Prospectus filed with the Securities and Exchange Commission.
Investing in Lending Club Notes involves risks, including the risk borrowers will not repay their loans and the risk of Lending Club discontinuing the servicing of the loans. Lending Club's obligation to make any payment on a Note is wholly dependent upon a borrower paying Lending Club on the corresponding loan in which you invested.
The risks of investing mean that investors may lose all or most of their investment. Before purchasing any of our Notes, you should carefully read our Prospectus, particularly the "Risk Factors" section on pages 17-32, which provides detailed information about the risks of investing in our Notes. The Notes are not guaranteed or insured by any governmental agency or instrumentality or any third party.
The Notes are presently being offered and sold solely to residents of the states of California, Colorado, Connecticut, Delaware, Florida, Georgia, Hawaii, Idaho, Illinois, Kentucky, Louisiana, Minnesota, Mississippi, Missouri, Montana, Nevada, New Hampshire, New York, Rhode Island, South Carolina, South Dakota, Virginia, Washington, West Virginia, Wisconsin, and Wyoming, and are not presently being offered or sold to residents of any other state, the District of Columbia, any other territory or possession of the United States, or any foreign country.
The Note Trading Platform operated by FOLIOfn is currently available to residents of all U.S. States except District of Columbia, Kansas, Maryland, Ohio, Oregon, and Vermont.
We file annual, quarterly and current reports and other information with the SEC. You may read and copy the registration statement for the Notes and any other documents we have filed at the SEC's Public Reference Room at 100 F Street, N.E., Room 1580, Washington, D.C. 20549. Please call the SEC at 1-800-SEC-0330 for further information on the Public Reference Room. Our SEC filings are also available to the public at http://www.sec.gov.

           """

title = "SVM Risk Consulting Loans for %s"% \
    datetime.now().date().strftime('%Y-%m-%d')

def send_mail(send_from, send_to, subject, text, files=[],
              #server="smtp.ezysurf.co.nz"
              ):
    print "sending email"
    assert type(send_to)==list
    assert type(files)==list

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach( MIMEText(text) )

    for f in files:
        part = MIMEBase('application', "octet-stream")
        part.set_payload( open(f,"rb").read() )
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(f))
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com',587) #port 465 or 587
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login('svmriskconsulting@gmail.com', "wpqqwtutulooanif")
    server.sendmail(send_from, send_to, msg.as_string())
    server.close()
