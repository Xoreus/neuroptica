#!/bin/bash
for FILE in ./*.pdf; do
  pdfcrop "${FILE}"
  mv "${FILE}" "Not_Cropped/${FILE}"
done
