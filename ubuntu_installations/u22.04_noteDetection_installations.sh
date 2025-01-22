cd
sudo apt update
sudo apt-get install pip
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt install curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std
sudo apt install git
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.8-venv python3.8-dev
python3.8 -m venv coralenv
echo "Entering Virtual Environment"
source coralenv/bin/activate
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
pip install --upgrade pip
git clone https://github.com/Team7520/EdgeTPU-Vision
cd EdgeTPU-Vision/
pip install mjpeg-streamer
pip install pynetworktables
python main.py --weights ~/EdgeTPU-Vision/model/edgetpu.tflite --labelmap ~/EdgeTPU-Vision/model/labelmap.pbtxt 
