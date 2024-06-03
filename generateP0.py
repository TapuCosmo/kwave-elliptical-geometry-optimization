import random
import math
import GenProcTrees as treeGen
from treeToImage import treeToImage
from PIL import Image, ImageDraw

def generateP0(imageSize = (1024, 1024), eccentricity = 0, ellipseWidth = 700, posOffset = (0, 0), rotOffset = 0, thicknessScale = 4, drawScale = 500, seed = None):
  ellipseHeight = ellipseWidth * math.sqrt(1 - eccentricity ** 2)

  if seed is not None:
    random.seed(seed)
  
  mask = Image.new("RGBA", imageSize, (0, 0, 0, 0))
  maskDraw = ImageDraw.Draw(mask)
  maskDraw.ellipse(
    [
      ((imageSize[0] - ellipseWidth) / 2, (imageSize[1] - ellipseHeight) / 2),
      ((imageSize[0] + ellipseWidth) / 2, (imageSize[1] + ellipseHeight) / 2)
    ],
    fill = (255, 255, 255, 255),
    width = 0
  )

  blankImage = Image.new("RGBA", imageSize, (0, 0, 0, 0))
  finalImage = Image.new("RGBA", imageSize, (255, 255, 255))
  
  treeImgs = []
  for i in range(3):
    tree = treeGen.generate_tree({
      'branch_length': 0.08,
      'turn_factor': 0.4,
      'leaf_start': 0.1,
      'min_distance': 0.1,
      'max_distance': 1,
      'number_of_leaves': 4000,
      'limit_2d': True
    })
    img = treeToImage(tree, imageSize = imageSize, thickness_scale = thicknessScale, draw_scale = drawScale).rotate(i * 120 + rotOffset, translate = posOffset)
    img = Image.composite(img, blankImage, mask)
    finalImage.alpha_composite(img)

  return finalImage.convert("1")