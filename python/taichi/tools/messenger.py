import atexit
import os
import smtplib
import socket

import taichi as tc

gmail_sender = 'taichi.messager@gmail.com'
gmail_passwd = '6:L+XbNOp^'

emailed = False


def send_crash_report(message, receiver=None):
    global emailed
    if emailed:
        return
    emailed = True
    if receiver is None:
        receiver = os.environ.get('TI_MONITOR_EMAIL', None)
    if receiver is None:
        tc.warn('No receiver in $TI_MONITOR_EMAIL')
        return
    tc.warn(f'Emailing {receiver}')
    TO = receiver
    SUBJECT = 'Report'
    TEXT = message

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(gmail_sender, gmail_passwd)

    BODY = '\r\n'.join([
        f'To: {TO}', f'From: {gmail_sender}', f'Subject: {SUBJECT}', '', TEXT
    ])

    try:
        server.sendmail(gmail_sender, [TO], BODY)
    except:
        print('Error sending mail')
    server.quit()
    print('Press enter or Ctrl + \\ to exit.')


def enable(task_name):
    register_call_back(task_name)


crashed = False
keep = []


def register_call_back(task_name):
    def at_exit():
        if not crashed:
            message = f'Congratulations! Your task [{task_name}] at machine [{socket.gethostname()}] has finished.'
            send_crash_report(message)

    def email_call_back(_):
        global crashed
        crashed = True
        tc.warn('Task has crashed.')
        message = f'Your task [{task_name}] at machine [{socket.gethostname()}] has crashed.'
        send_crash_report(message)
        atexit.unregister(at_exit)
        exit(-1)

    keep.append(email_call_back)
    # TODO: email_call_back should be passed to Taichi core (C++). It will then called by the signal handler when Taichi crashes
    # (std::function<void(int)> python_at_exit)
    # Simply register a callback in the Python scope will not work in cases when Taichi crashes
    # call_back = tc.function11(email_call_back)
    # tc.core.register_at_exit(call_back)

    atexit.register(at_exit)


if __name__ == '__main__':
    register_call_back('test')
    tc.core.trigger_sig_fpe()
