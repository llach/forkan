import os


class CSVLogger(object):

    def __init__(self, path, *labels):
        assert '.csv' in path, 'give full csv file path to write to'

        self.path = path
        self.labels = list(labels)

        for i, _ in enumerate(self.labels):
            self.labels[i] = str(self.labels[i])

        # file handle for writing episode summaries
        self.fh = open('{}'.format(self.path), 'a')

        # write headline if file is empty
        if os.stat('{}'.format(self.path)).st_size == 0:
            self.fh.write('{}\n'.format(', '.join(self.labels)))

    def __del__(self):

        # flush internal buffer and close filehandle
        if hasattr(self, 'fh'):
            self.fh.flush()
            self.fh.close()

    def writeline(self, *values):
        assert len(values) == len(self.labels), 'trying to set {} entries for {} labels'.format(len(values),
                                                                                                len(self.labels))
        values = list(values)
        for i, _ in enumerate(values):
            values[i] = str(values[i])
        self.fh.write('{}\n'.format(', '.join(values)))
