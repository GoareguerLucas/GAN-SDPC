default: rsync_image

start_ssh:
	sudo systemctl start ssh

stop_ssh:
	sudo systemctl stop ssh

connect_ssh:
	ssh albert@147.94.234.179

connect_gt2:
	ssh -p8012 g14006889@gt-2.luminy.univ-amu.fr

connect_gt0:
	ssh -p8012 g14006889@gt-0.luminy.univ-amu.fr

rsync_image:
	rsync -av . albert@147.94.234.179:~/Bureau/Lucas/GAN-SDPC/ --exclude-from=ExclusionRSync