from qiskit.primitives import BackendSamplerV2, BackendEstimatorV2
from qiskit.providers import JobStatus
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeJobV2, SamplerV2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from random import randint
from math import ceil

from os import listdir, name
from os.path import isfile, join
PATH = 'C:/qiskit/'
# This is for the system it was designed on, include your own path

from warnings import warn

from matplotlib import pyplot as plt
plt.style.use('ggplot')


def authenticate():
    file = open(PATH+"api", "r")
    api = file.readline()[0:-1]
    file.close()
    file = open(PATH+"crn", "r")
    crn = file.readline()[0:-1]
    file.close()
    QiskitRuntimeService.save_account(token=api, instance=crn, plans_preference=["open"], overwrite=True)

def generate_dict(size):
    generator = [["0" for j in range(size)] for i in range(2**size)]
    keys = ["" for i in range(2**size)]
    for i in range(2**size):
        number = i
        for j in range(size-1,-1,-1):
            if number-(2**j) >= 0:
                generator[i][size-1-j] = "1"
                number -= 2 ** j
        for j in range(size):
            keys[i]+=generator[i][j]
    dictionary = {}
    for key in keys:
        dictionary[key] = 0
    return dictionary

def get_probs(result, dictionary, percentage = False):
    data = result[0].data.meas.get_counts()
    for key in data:
        dictionary[key] = data[key] / (percentage*result[0].data.meas.num_shots+(1-percentage))
    return dictionary

def tidy_probs(dictionary, relevance=0.005):
    keys=list(dictionary.keys())
    output_dict = {}
    for key in keys:
        if dictionary[key] > relevance:
            output_dict[key]=dictionary[key]
    return output_dict

def get_expectation_values(result, observables_labels):
    output = {}
    for i in range(len(observables_labels)):
        output[observables_labels[i]] = result[0].data.evs[i]
    return output

def draw_bar(x,y,xlabel=None,ylabel=None,title=None):
    plt.bar(x, y)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def draw_plot(x,y,xlabel=None,ylabel=None,title=None):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

class Service(QiskitRuntimeService):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.known_jobs = {}
        self.update_jobs()

    def update_jobs(self):
        """Updates internal list of jobs.

        Returns:
            self.known_jobs (dict)
        """
        job_path = PATH+"jobs/"
        self.known_jobs = {}
        current_jobs = self.jobs()
        saved_jobs = [f for f in listdir(job_path) if isfile(join(job_path, f))]
        for job in saved_jobs:
            file = open(PATH+"jobs/" + job, "r")
            self.known_jobs[job] = self.job(file.read())
            file.close()
        for job in current_jobs:
            check = True
            for existing_job in list(self.known_jobs.values()):
                if existing_job.job_id() == job.job_id():
                    check = False
                    break
            if check: self.known_jobs[job.job_id()] = job
        return self.known_jobs

    def cancel_all(self):
        """Cancels all jobs with status INITIALIZING or QUEUED"""
        self.update_jobs()
        # "INITIALIZING", "QUEUED", "RUNNING", "CANCELLED", "DONE", "ERROR"
        for job in self.known_jobs.values():
            if job.status() in ["INITIALIZING", "QUEUED"]:
                job.cancel()

    def get_unnamed(self):
        """Returns a dictionary of all jobs that do not have a name saved"""
        self.update_jobs()
        unnamed = {}
        # "INITIALIZING", "QUEUED", "RUNNING", "CANCELLED", "DONE", "ERROR"
        for key in self.known_jobs.keys():
            job = self.known_jobs[key]
            if key == job.job_id():
                unnamed[key] = job
        return unnamed

class QPU:
    def __init__(self, service = None, backend = None):
        if service is None: self.service = Service()
        else: self.service = service
        if backend is None: self.backend = service.least_busy(simulator=False, operational=True)
        else: self.backend = backend
        self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=3)
        self.estimator = BackendEstimatorV2(backend = self.backend)
        self.config_estimator()
        self.sampler = BackendSamplerV2(backend = self.backend)
        self.isa_circuit = None
        self.known_jobs = {}
        self.update_jobs()

    def config_estimator(self, resilience_level=1, default_shots=5000):
        """Updates the estimator configuration."""
        self.estimator.options.resilience_level = resilience_level
        self.estimator.options.default_shots = default_shots

    def update_isa_circuit(self, qc):
        self.isa_circuit = self.pm.run(qc)

    def run_sampler(self, qc, shots=1024):
        return self.sampler.run([self.isa_circuit],shots=shots)

    def run_estimator(self, qc, observables, mapped_observables = False):
        if not mapped_observables:
            mapped_observables = [observable.apply_layout(self.isa_circuit.layout) for observable in observables]
        else:
            mapped_observables = observables
        return self.estimator.run([(self.isa_circuit, mapped_observables)])

    def update_jobs(self):
        """
        Updates internal list of jobs.

        Returns:
            self.known_jobs (dict)
        """
        job_path = PATH+"jobs/"
        self.known_jobs = {}
        current_jobs = self.service.jobs()
        saved_jobs = [f for f in listdir(job_path) if isfile(join(job_path, f))]
        for job in saved_jobs:
            file = open(PATH+"jobs/" + job, "r")
            self.known_jobs[job] = self.service.job(file.read())
            file.close()
        for job in current_jobs:
            check = True
            for existing_job in list(self.known_jobs.values()):
                if existing_job.job_id() == job.job_id():
                    check = False
                    break
            if check: self.known_jobs[job.job_id()] = job
        return self.known_jobs

    def get_status(self, job):
        """Retrieve the status of a given job.

        Args:
            job: (str or Job) the job to retrieve the status of.

        Returns:
            job.status().

        Raises:
            Exception: Unknown error when trying to retrieve job status
        """
        job = self.get_job(job)
        try:
            return job.status()
        except Exception as e:
            raise e

    def get_job(self, job):
        """Retrieve Job object for an existing job with either ID or name.

        Args:
            job: if type(str) returns self.known_jobs[job], else if type(Job) returns job.

        Returns:
            Job object if a job with the given name/ID is found, else None

        Raises:
            KeyError: if the job is in the known job keys, but cannot be accessed
            Exception: Unknown error when trying to retrieve job
        """
        self.update_jobs()
        if type(job) == RuntimeJobV2: return job
        elif type(job) != str: raise f'arg "job" must be of type {type(self.known_jobs[list(self.known_jobs.keys())[0]])} or str'
        if job in self.known_jobs.keys():
            try:
                return self.known_jobs[job]
            except KeyError:
                raise f'Job {job} is in the known jobs keys, but cannot be accessed'
            except Exception as e:
                raise e
        return None

    def get_result(self, job):
        """Retrieve the result for a given job.

        Args:
            job: (str or Job) the job to retrieve the result for.

        Returns:
            job.result() if finished, True if the job is still running, False if the job has been cancelled or errored, None if the job does not exist.

        Raises:
            Exception: Unknown error when trying to retrieve job result
        """
        job = self.get_job(job)
        if job is None: return None
        status = job.status()
        # "INITIALIZING", "QUEUED", "RUNNING", "CANCELLED", "DONE", "ERROR"
        if status == "DONE" or status == JobStatus.DONE:
            return job.result()
        elif status in ["INITIALIZING", "QUEUED", "RUNNING"] or status in [JobStatus.INITIALIZING, JobStatus.QUEUED, JobStatus.RUNNING]:
            return True
        elif status in ["CANCELLED","ERROR"] or status in [JobStatus.CANCELLED, JobStatus.ERROR]:
            return False
        else:
            raise Exception("Unknown error retrieving result")

    def save_job(self, job, name):
        """
        Saves the job ID under the given name.

        :param job: (str or Job) the job to save the ID of.
        :param name: (str) the name of the job.

        :return: A string containing the location of the file to which the job was saved, or None if a job with that name already exists, in which case the new job will be cancelled
        """
        save_location = PATH+"jobs/" + name
        if self.get_job(name):
            job.cancel()
            warn("A job with this name already exists and as such the new job has been cancelled")
            return None
        try:
            file = open(save_location, "r")
            file.close()
            job.cancel()
            warn("A job with this name already exists and as such the new job has been cancelled")
            return None
        except FileNotFoundError:
            file = open(save_location, "w")
            file.write(job.job_id())
            file.close()
            self.update_jobs()
            return save_location
        except Exception as e:
            raise e

class Simulation(QPU):
    def __init__(self, service=None, backend=None):
        service = service if service else False
        backend = backend if backend else AerSimulator()
        super().__init__(service, backend)

    def update_jobs(self):
        return self.known_jobs

    def save_job(self, job, name):
        self.known_jobs[name] = job



if __name__ == '__main__':
    service = False
    check1 = input("If you would like to reauthenticate type 'YES': ")
    check2 = input("If you would like to check current jobs type 'YES': ")
    check3 = input("If you would like a list of unnamed jobs type 'YES': ")
    if check1=='YES':
        authenticate()
    if check2=='YES':
        service = Service()
        print(service.known_jobs.keys())
    if check3=='YES':
        if not service:service = Service()
        service.get_unnamed()