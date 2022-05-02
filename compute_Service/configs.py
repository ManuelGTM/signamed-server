import yaml
class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as ymlfile:
            cfg = yaml.load(ymlfile)
            
        self.ip_server_compute = cfg['IP_SERVER_COMPUTE']
        self.msg3d_configs_file = cfg['MSG3D_CONFIG_FILE'] 
        self.channels = cfg['CHANNELS']
        self.frames = cfg['FRAMES']
        self.num_kps = cfg['NUM_KPS']

    def __str__(self):
        return 'ip_server_compute={}msg3d_configs_file={}channels={}frames-anns={}&num_kps={}'.format(
            self.ip_server_compute,
            self.msg3d_configs_file,
            self.channels,
            self.frames,
            self.num_kps
        )