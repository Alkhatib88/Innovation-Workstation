from import_modules import *

class App:
    def __init__(self):
        self.logger = Logger(self)
        self.error_handler = ErrorHandler(self)
        self.system_info = SystemInfo(self)
        self.cli = Cli(self)
        self.cli_tools = CliTools(self)
        self.command = Command(self)
        self.input = Input_sys(self, CustomError, DependencyError, UnknownCommandError)
        self.file_opt = FileOperations(self)
        self.dir_opt = DirectoryOperations(self)
        self.sql = PostgreSQL(self)
        self.encrypt = EncryptionSystem(self)

        #self.config = ConfigSystem(self)
        #self.env = EnhancedEnvironmentVariables(self)
        #self.pickel = PickleStorage(self)
        



    def setup(self,log_file=None):
        """Sets up the application components."""
        if log_file is None:
            # Default to logs folder in the current directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            log_file = os.path.join(base_dir, 'logs', 'app.log')

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.logger.setup(name='AppLogger', log_file=log_file, level=logging.DEBUG)

        self.system_info.setup_commands()
        self.sql.setup()
        self.cli_tools.setup()
        self.cli.setup()
        self.file_opt.setup()
        self.dir_opt.setup()



        '''
        # Setup ConfigSystem for different parts of the app
        self.config.setup('default', '/home/coder/python_project/config/default_config', 'ini')
        self.env.setup('main', self.env_file_path, 'file')

        try:
            # Initialize primary encryption key
            self.encrypt.initialize_encryption_keys()
            # Retrieve and/or generate the secondary encryption key
            second_encryption_key = self.encrypt.sec_encrypt_key(self.env_file_path)
            # Use the second encryption key for further operations
            self.env.setup('main',self.env_file_path,'file')


        except Exception as e:
            # Handle any exception that occurs during setup
            self.logger.info(f'{e}')
            self.error_handler.handle_error(e)

        '''

    def run(self):


        self.cli.run()  

if __name__ == "__main__":
    app = App()
    app.setup()
    app.run()