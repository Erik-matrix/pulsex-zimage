@echo off
rem zimage - gor en bild ur en prompt. Alla argument gar vidare till zimage.py.
rem   zimage "a photo of a cat" -o katt.jpg --size 512
python "%~dp0zimage.py" %*
