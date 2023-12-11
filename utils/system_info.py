import os
import psutil
import platform
import subprocess

class SystemInfo:
    def __init__(self, app):
        self.app = app

    def get_cpu_info(self):
        cpu_info = {
            "Physical cores": psutil.cpu_count(logical=False),
            "Total cores": psutil.cpu_count(logical=True),
            "Max Frequency": f"{psutil.cpu_freq().max:.2f}Mhz",
            "Min Frequency": f"{psutil.cpu_freq().min:.2f}Mhz",
            "Current Frequency": f"{psutil.cpu_freq().current:.2f}Mhz",
            "CPU Usage (over 1 sec)": f"{psutil.cpu_percent(interval=1)}%",
        }
        return cpu_info

    def get_ram_info(self):
        svmem = psutil.virtual_memory()
        ram_info = {
            "Total RAM": f"{svmem.total / (1024 ** 3):.2f} GB",
            "Available RAM": f"{svmem.available / (1024 ** 3):.2f} GB",
            "Used RAM": f"{svmem.used / (1024 ** 3):.2f} GB",
            "RAM Percentage": f"{svmem.percent}%",
        }
        return ram_info

    def get_storage_info(self):
        storage_info = {}
        try:
            if platform.system() == "Windows":
                result = os.popen('wmic logicaldisk where "drivetype=3" get caption, freespace, size').read().split("\n")[1:-1]
                for line in result:
                    if line.strip():
                        parts = line.split()
                        drive, free_space, total_space = parts[0], int(parts[1]) / (1024 ** 3), int(parts[2]) / (1024 ** 3)
                        used_space = total_space - free_space
                        percent_used = (used_space / total_space) * 100
                        storage_info[drive] = {
                            "Total Size": f"{total_space:.2f} GB",
                            "Used Size": f"{used_space:.2f} GB",
                            "Free Size": f"{free_space:.2f} GB",
                            "Percentage Used": f"{percent_used:.2f}%",
                        }

            elif platform.system() == "Linux":
                result = subprocess.check_output(['df', '-h']).decode().split("\n")
                for line in result[1:]:
                    if line:
                        parts = line.split()
                        if '/dev/' in parts[0]:
                            drive, total_space, used_space, free_space, percent_used = parts[0], parts[1], parts[2], parts[3], parts[4]
                            storage_info[drive] = {
                                "Total Size": total_space,
                                "Used Size": used_space,
                                "Free Size": free_space,
                                "Percentage Used": percent_used,
                            }

        except Exception as e:
            storage_info["Error"] = f"Failed to retrieve information due to: {e}"
        return storage_info

    def get_network_info(self):
        net_io = psutil.net_io_counters()
        network_info = {
            "Bytes Sent": f"{net_io.bytes_sent / (1024 ** 2):.2f} MB",
            "Bytes Received": f"{net_io.bytes_recv / (1024 ** 2):.2f} MB",
        }
        return network_info

    def get_running_apps_info(self):
        running_apps = {}
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            pid = proc.info['pid']
            name = proc.info['name']
            cpu_usage = proc.info['cpu_percent']
            mem_usage = proc.info['memory_percent']
            
            running_apps[name] = {
                "PID": pid,
                "CPU Usage": f"{cpu_usage}%",
                "Memory Usage": f"{mem_usage:.2f}%",
            }
        return running_apps

    def log_all_info(self):
        try:
            self.app.logger.info("Fetching system information")

            cpu_info = self.get_cpu_info()
            self.app.logger.info("CPU Information:")
            for key, value in cpu_info.items():
                self.app.logger.info(f"    {key}: {value}")

            ram_info = self.get_ram_info()
            self.app.logger.info("RAM Information:")
            for key, value in ram_info.items():
                self.app.logger.info(f"    {key}: {value}")

            storage_info = self.get_storage_info()
            self.app.logger.info("Storage Information:")
            for drive, info in storage_info.items():
                self.app.logger.info(f"    Drive {drive}:")
                for key, value in info.items():
                    self.app.logger.info(f"        {key}: {value}")

            network_info = self.get_network_info()
            self.app.logger.info("Network Information:")
            for key, value in network_info.items():
                self.app.logger.info(f"    {key}: {value}")

            # Given the potential length of the running apps, you may want to comment this out or only log a subset.
            #turn this into a command
            running_apps_info = self.get_running_apps_info()
            self.app.logger.info("Running Applications:")
            for app, info in running_apps_info.items():
                self.app.logger.info(f"    {app}:")
                for key, value in info.items():
                    self.app.logger.info(f"        {key}: {value}")

        except Exception as e:
            self.app.error_handler.handle_error(e)

    def setup_commands(self):
        self.app.command.add_command(command_name="get-cpu-info", function=self.get_cpu_info, description="Display CPU information", category="System Diagnostics")
        self.app.command.add_command(command_name="get-ram-info", function=self.get_ram_info, description="Display RAM information", category="System Diagnostics")
        self.app.command.add_command(command_name="get-storage-info", function=self.get_storage_info, description="Display Storage information", category="System Diagnostics")
        self.app.command.add_command(command_name="get-network-info", function=self.get_network_info, description="Display Network information", category="System Diagnostics")
        self.app.command.add_command(command_name="task-manager", function=self.get_running_apps_info, description="Displays current running apps", category="System Diagnostics")