import collections
from multiprocessing import Process, Manager


class SimScheduler:

    def __init__(self):
        self.manager = Manager()
        self.processes = [] # type: list[Process]
        self.return_obj = self.manager.dict()

    def add_simulation(self, func, **params):
        """
        Execute script with input parameters, retrieve output
        :param exec: script to run
        :param params: input parameters
        :return: output of the script
        """
        params["return"] = self.return_obj

        p = Process(target=func, kwargs=params)
        self.processes.append(p)

    def start_all(self, chunk_size=4):

        # print("starting processes")
        chunks = [
            self.processes[i:i + chunk_size]
            for i in range(0, len(self.processes), chunk_size)
        ]

        i=1
        for chunk in chunks:
            print('###chunk {} of {}###'.format(i, len(chunks)))
            for p in chunk:
                p.start()

            # print("waiting to finish")
            for p in chunk:
                p.join()
            i+=1
        return [x for x in collections.OrderedDict(sorted(self.return_obj.items())).values()]
