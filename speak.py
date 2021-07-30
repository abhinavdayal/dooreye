class Speak():
    def emit(self, msg):
        cmd = ['espeak', '-ven+f3', '-s150', msg]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        p.communicate()

def configure_logging():
    h = TalkHandler()
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)

def main():
    logging.info('Hello')
    logging.debug('Goodbye')

if __name__ == '__main__':
    configure_logging()
    sys.exit(main())