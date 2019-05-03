default: rsync_image

clean_models:
	rm */models/*.pt

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

matplot:
	export MPLCONFIGDIR="/var/lib/vz/data/g14006889/cache/matplotlib/"

scp_gt0:
	scp -P 8012 -r g14006889@gt-0.luminy.univ-amu.fr:/var/lib//vz/data/g14006889/GAN-SDPC/W3_current_began/M_k.png .

weigth:
	du -h . --max-depth=1

see_rsync:
	rsync -avhuzn . albert@147.94.234.179:~/Bureau/Lucas/GAN-SDPC/ --exclude-from=ExclusionRSync

rsync:
	rsync -avhuz . albert@147.94.234.179:~/Bureau/Lucas/GAN-SDPC/ --exclude-from=ExclusionRSync
