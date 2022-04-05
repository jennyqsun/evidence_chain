
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:00:45 2022

@author: jenny
"""


import serial
import sys
import numpy as np
import glob
import struct




def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/ttyUSB*')  # ubuntu is /dev/ttyUSB0
    elif sys.platform.startswith('darwin'):
        # ports = glob.glob('/dev/tty.*')
        ports = glob.glob('/dev/tty.SLAB_USBtoUART*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except serial.SerialException as e:
            if e.errno == 13:
                raise e
            pass
        except OSError:
            pass
    return result


def send_ser_command(device, command, bytes_expected=0):
    device.write(command)
    response = device.read(bytes_expected)

    return response



def identiy_device():
    portname = serial_ports()[0]
    with serial.Serial(portname, 115200, timeout=1) as ser:
        ser.write(b"_c1")  # byte string
        query_return = ser.read(5)
        print('device response: ', query_return)

    if (len(query_return)) > 0 & (query_return == b"_xid0"):
        print('device detected!')



def get_model(portname):
 # identify which device we're speaing to
    with serial.Serial(portname, 115200, timeout=1) as ser:
        ser.write(b"_d2")  # byte string
        device_id = ser.read(1)

        ser.write(b"_d3")  # byte string
        model_id = ser.read(1)
    return device_id, model_id




def decimalToBinary(n):
    return bin(n).replace("0b", "")

def def_keyboard(device_id, model_id):
    rb_540_keymap = [-1, 0, -1, 1, 2, 3, 4, -1]
    rb_740_keymap = [-1, 0, 1, 2, 3, 4, 5, 6]
    rb_840_keymap = [7, 3, 4, 1, 2, 5, 6, 0]
    rb_834_keymap = [7, 0, 1, 2, 3, 4, 5, 6]
    lumina_keymap = [-1, 0, 1, 2, 3, 4, -1, -1]
    keymap = []

    if device_id == b'2':
        if model_id == b'1':
            keymap = rb_540_keymap
            print('model: rb_540_keymap' )
        elif model_id == b'2':
            keymap = rb_740_keymap
            print('model: rb_740_keymap' )
        elif model_id == b'3':
            keymap = rb_840_keymap
            print('model: rb_840_keymap' )
        elif model_id == b'4':
            keymap = rb_834_keymap
            print('model: rb_834_keymap' )
        else:
            print("An unknown RB model was detected. Very strange.")
            keymap == []
    if keymap:
        print('keymap found')
    else:
        print('no keymap')
    return keymap



def getKey(portname, keymap,timeout):
    resp = []
    respInfo = []
    key = []
    pressed = []
    s = serial.Serial(portname, 115200, timeout=timeout) 
    s.write(b'e1')   #reset RT and base timer
    s.write(b'e5')
    k = s.read(6)
            # print(k)
    if not k:
        pass
    else:
        resp.append(k)
        respInfo = list(decimalToBinary(resp[0][1]))
        respInfo = [(int(i)) for i in respInfo]
        respInfo = np.pad(respInfo, (8-len(respInfo), 0))
        key = respInfo[2] + respInfo[1]*2 + respInfo[0]*4
        # port = respInfo[7] + respInfo[6]*2 + respInfo[5]*4 + respInfo[4]*8
        pressed = respInfo[3]
        key = keymap[key]
        stamp = resp[0][2:6]

    return resp, key, pressed, stamp







def getKeypress(portname, keymap, timeout, expectkeys):
    resp = []
    respInfo = []
    key = []
    pressed = []
    s = serial.Serial(portname, 115200, timeout=timeout) 
    s.write(b'e1')   #reset RT and base timer
    s.write(b'e5')
    k = s.read(6*expectkeys*2)
            # print(k)
    if not k:
        pass
    else:
        resp.append(k)
        # respInfo = list(decimalToBinary(resp[0][1]))
        # respInfo = [(int(i)) for i in respInfo]
        # respInfo = np.pad(respInfo, (8-len(respInfo), 0))
        # key = respInfo[2] + respInfo[1]*2 + respInfo[0]*4
        # # port = respInfo[7] + respInfo[6]*2 + respInfo[5]*4 + respInfo[4]*8
        # pressed = respInfo[3]
        # key = keymap[key]
        # stamp = resp[0][2:6]

    return k



def parseoutput(k):
    # parse the responses
    resp = []
    count = 0
    for i in range(len(k)):
        if i%6==0:
            resp.append(k[count: count+6])
            count = count + 6
    return resp

def readoutput(resp, keymap):            
    key = []
    press=[]
    stamp = []

    for r in resp:
        respInfo = list(decimalToBinary(r[1]))
        respInfo = [(int(i)) for i in respInfo]
        respInfo = np.pad(respInfo, (8-len(respInfo), 0))
        key_ = respInfo[2] + respInfo[1]*2 + respInfo[0]*4
        # port = respInfo[7] + respInfo[6]*2 + respInfo[5]*4 + respInfo[4]*8
        press.append(respInfo[3])
        key.append(keymap[key_])
        stamp.append(r[2:6])
    return key, press, stamp


def getresponse(portname, keymap, timeout, expectkeys):
    print('start pressing the keys')
    k = getKeypress(portname,keymap, timeout, expectkeys)
    resp = parseoutput(k)
    out = readoutput(resp)
    return resp,out


def BytesListToHexList(tlist):
    '''input:  a list of 4 bytes array
       return: a list of hex (each has 2*4 bits) '''
    th = [t.hex() for t in tlist]
    return th 

def HexToRt(t):
    '''input: 2 x 4 bits of hex
    return :  endian swap and convert into decimal'''
    if type(t) == list:
        t = t[0]
    rt = [t[i:i+2] for i,j in enumerate(t) if i%2 ==0 ][::-1]
    rt = ''.join(map(str,rt))
    rt = int(rt,16)
    return rt
# set up resposne pad

def getname():
    
    portname = serial_ports()[0]
    print('writing to device...')
    identiy_device()
    device_id, model_id = get_model(portname)
    keymap = def_keyboard(device_id, model_id)
    return portname, keymap



def reset_timer(s):
    s.write(b'e1')   #reset RT and base timer
    s.write(b'e5')

def clear_buffer(s):
    s.reset_input_buffer()
    s.reset_output_buffer()

# portname, keymap = getname()
# out = getresponse(portname=portname, keymap=keymap, timeout = None, expectkeys = 1)


# you can change timeout and ecpectkeys depending on how you want to integrate to your task
# when expectkeys is 1, serial port sends 

