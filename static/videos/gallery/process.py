import os
import tqdm
def changecodec(vid):
  cmd = f"""ffmpeg -i raw/{vid} -c:v libx264 -crf 23 -preset fast -c:a aac -b:a 192k codec/{vid} """
  os.system(cmd)


def webify():
  videos = [f for f in os.listdir('codec') if '.mp4' in f]
  print(videos)
  print(len(videos))
  for i in range(len(videos)):
    if i % 2 == 1:
      continue
    vid1 = videos[i]
    vid2 = videos[i+1]
    print(f"""
<div class="item item-fullbody">
  <!-- {vid1} -->
  <video poster="" id="fullbody" autoplay playsinline muted loop height="450px" width="auto">
    <source src="static/videos/gallery/codec/{vid1}" type="video/mp4">
  </video>
<br>
  <!-- {vid2} -->
  <video poster="" id="fullbody" autoplay playsinline muted loop height="288px">
    <source src="static/videos/gallery/codec/{vid2}" type="video/mp4">
  </video>
</div>
  """)

if __name__=='__main__':

  if 0:
    videos = os.listdir('raw')
    os.makedirs('codec', exist_ok=True)
    print(len(videos))
    for vid in videos:
      changecodec(vid)

  if 1:
    webify()

