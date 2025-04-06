import numpy as np
import win32gui, win32ui, win32con, win32api
from contextlib import contextmanager

@contextmanager
def win32_resources():
    resources = []
    try:
        yield resources
    finally:
        for resource_type, resource in reversed(resources):
            if resource_type == 'DC':
                resource.DeleteDC()
            elif resource_type == 'handle':
                win32gui.ReleaseDC(*resource)
            elif resource_type == 'object':
                win32gui.DeleteObject(resource)

def screen_capture(region=None):
    hwin = win32gui.GetDesktopWindow()
    if region:
        left, top, right, bottom = region
        width = right - left + 1
        height = bottom - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
    
    with win32_resources() as resources:
        hwindc = win32gui.GetWindowDC(hwin)
        resources.append(('handle', (hwin, hwindc)))
        
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        resources.append(('DC', srcdc))
        
        memdc = srcdc.CreateCompatibleDC()
        resources.append(('DC', memdc))
        
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        resources.append(('object', bmp.GetHandle()))
        
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img = img.reshape(height, width, 4)
        
        return img.copy()