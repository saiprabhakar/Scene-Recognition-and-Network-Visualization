from visuModels_sac import ex
from sacred.observers import FileStorageObserver

ex.observers = [FileStorageObserver.create('analysis_1')]
ex.run()

ex.observers = [FileStorageObserver.create('analysis_1')]
ex.run(config_updates={'heat_map_ratio': 0.75})
