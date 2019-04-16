default: rsync_image

start_ssh:
	sudo systemctl start ssh

stop_ssh:
	sudo systemctl stop ssh

connect_ssh:
	ssh albert@147.94.234.179

rsync_image:
	rsync -av . albert@147.94.234.179:~/Bureau/Lucas/GAN-SDPC/ --exclude-from=ExclusionRSync