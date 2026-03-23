# 1. 在 data 文件夹内为 CARLA 创建专属目录
mkdir -p ~/data/carla0915
cd ~/data/carla0915

# 2. 下载主程序包（约 15-20GB，取决于网络速度）
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz

# 3. 解压文件（这一步会比较慢，请耐心等待）
tar -xvf CARLA_0.9.15.tar.gz

# 4. 删除压缩包以节省空间
rm CARLA_0.9.15.tar.gz

# 5. 下载并导入额外地图（Additional Maps）
cd Import
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. 
bash ImportAssets.sh
