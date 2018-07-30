# Python program to illustrate HoughLine
# method for line detection
import cv2
import numpy as np

from scipy import ndimage



def rotateImage(image, angle):
  '''
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  '''
 #rotation angle in degree

  rotated = ndimage.rotate(image, angle=angle)

  return rotated
def linedet(img,out):
  h, w, t = img.shape
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
  for line in lines:
      for r, theta in line:
          # Stores the value of cos(theta) in a
          a = np.cos(theta)

          # Stores the value of sin(theta) in b
          b = np.sin(theta)

          # x0 stores the value rcos(theta)
          x0 = a * r

          # y0 stores the value rsin(theta)
          y0 = b * r

          # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
          x1 = int(x0 + 1000 * (-b))

          # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
          y1 = int(y0 + 1000 * (a))

          # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
          x2 = int(x0 - 1000 * (-b))

          # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
          y2 = int(y0 - 1000 * (a))

          # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
          # (0,0,255) denotes the colour of the line to be
          # drawn. In this case, it is red.

          #cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
          #print x1, y1, x2, y2

          #if((abs(x1-x2)<300 and abs(x1-x2)>200) or (abs(y1-y2)<300 and abs(y1-y2)>200)):
          if ((float(y1)/float(h)>.87 and float(y1)/float(h)<.90) or (float(y1)/float(h)<.12 and float(y1)/float(h)>.10) ):
              qwe=cv2.imread('image')

              cv2.line(qwe, (x1, y1), (x2, y2), (0, 0, 255), 2)
              if (abs(float(y1))/float(h)>.87 and abs(float(y1))/float(h)<.90):
                  crop_img = rotateImage(img, 180)
                  crop_img = crop_img[0:h - y2 + 10, 0:]
                  # cv2.imshow("cropped", crop_img)
                  return crop_img



              else:
                  crop_img = img[0:y2 + 10, 0:]
                  # cv2.imshow("cropped", crop_img)
                  return crop_img






# Reading the required image in
# which operations are to be done.
# Make sure that the image is in the same
# directory in which this python program is

def main(img):
  h,w,t=img.shape
  print(h,w,t)
  if h<w:
    img=rotateImage(img,270)

  h,w,t=img.shape
#    print h,w,t
  img=linedet(img,"out")
#    print x1,y1,x2,y2
  return img

