sudo swapoff -a
# size = bs*count Bytes, currently 8GB
#if test -f "/swapfile"; then
#	echo "swapfile already exists";
#else
	sudo dd if=/dev/zero of=/swapfile bs=1024 count=8000000 status=progress
	sudo chmod 0600 /swapfile
#fi
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon --show
