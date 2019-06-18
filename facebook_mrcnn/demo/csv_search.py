import os

'''
3가지 CSV 파일(dicom, patient, diagnosis)을 다루기 위한 제너레이터
각 파일의 주요 키가 일치하는 항목끼리 합쳐서 장별 또는 건별로 순회가 가능 

#예제

# 장별 데이터 순회
for row in csv_search.per_dicom():
    print(row)

# 장별 데이터 순회(조건 입력)
for row in csv_search.per_dicom(search={'Diagnosis': 'normal'}):
    print(row)

# 특정 조건에 맞는 데이터 갯수
print(len(list(csv_search.per_patient(search={'Date': '180718'}, include_excluded=True))))

# 특정 조건에 맞는 건별, 장별 순회
for d, l in csv_search.Per_patient_fast(search={'Date': '180718', 'Diagnosis': 'ckd'}):
    print('건당 정보', d)
    for t in l:
        print('장별 Dicom 정보', t)

'''

DEFAULT_INIT_DATA = None


def norm_path(path):
    path = os.path.normpath(path)
    path = os.path.normcase(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path


# compose dict data with header from csv
def load_dict(csv_file, key_name):
    d = dict()
    with open(csv_file) as fcsv:
        header = fcsv.readline().strip('\r\n').split(',')
        d['Header'] = header
        key_index = d['Header'].index(key_name)
        while True:
            line = fcsv.readline()
            if not line:
                break
            cols = line.strip('\r\n').split(',')
            d[cols[key_index]] = cols

        print('--load_dict header', header, "cnt_rows", len(d))
    return d


# read dict, each_key: each '' when key is not exists
def read_dict(dst_dict, key):
    d = dict()

    if key in dst_dict.keys():
        vl = dst_dict[key]
        for k, v in zip(dst_dict['Header'], vl):
            d[k] = v
    else:
        v = ''
        for k in dst_dict['Header']:
            d[k] = v

    return d


class Csv_data():
    def __init__(self,
                 diagnosis_csv_path='~/data/yonsei2/doc/진단정보/diagnosis_info_400+100+1200.csv',
                 dicom_csv_path='~/data/yonsei2/doc/Dicom정보/dicom_info_400+100+1200.csv',
                 patient_csv_path='~/data/yonsei2/doc/환자 정보/patient_info_400+100+1200.csv'):
        diagnosis_csv_path = norm_path(diagnosis_csv_path)
        dicom_csv_path = norm_path(dicom_csv_path)
        patient_csv_path = norm_path(patient_csv_path)

        # read Diagnosis
        self.DIAGNOSIS_Dict = load_dict(diagnosis_csv_path, 'File')
        self.DICOM_Dict = load_dict(dicom_csv_path, 'Name')
        self.PATIENT_Dict = load_dict(patient_csv_path, 'AccNo')


# 장별 제너레이터
class Per_dicom():

    def __init__(self, init_data=None, include_excluded=False, search={}):
        # read csv phase only onece
        if init_data:
            self.data = init_data
        else:
            global DEFAULT_INIT_DATA
            self.data = DEFAULT_INIT_DATA

        self.search = search
        self.include_excluded = include_excluded

        self.DIAGNOSIS_Dict = self.data.DIAGNOSIS_Dict
        self.DICOM_Dict = self.data.DICOM_Dict
        self.PATIENT_Dict = self.data.PATIENT_Dict

    def __iter__(self):

        for key_diagnosis in self.DIAGNOSIS_Dict.keys():

            if key_diagnosis == 'Header':
                continue

            d = dict()

            # merge dict element
            d.update(read_dict(self.DIAGNOSIS_Dict, key_diagnosis))

            key_dicom = key_diagnosis
            d.update(read_dict(self.DICOM_Dict, key_dicom))

            key_patient = d['AccNo']
            d.update(read_dict(self.PATIENT_Dict, key_patient))

            if self.include_excluded == False:
                if d['Excluded'] != '':
                    continue

            # match search
            match = True
            for search_key, search_value in self.search.items():
                if search_key in d.keys():
                    if d[search_key].lower() != search_value.lower():
                        match = False
                        break
                else:
                    print('not found search key', search_key)
                    raise KeyError

            # return or next
            if match:
                yield d
            else:
                continue


# 건별 제너레이터
class Per_patient_raw():

    def __init__(self, init_data=None, include_excluded=False, search={}):
        self.per_dicom = Per_dicom(init_data=init_data, include_excluded=include_excluded, search=search)
        self.returned_patient = set()
        self.patient_key_name = 'AccNo'
        self.pass_patient_keys = set(
            ['AccNo', 'RecoredPatientID', 'RealPatientID',
             'Date', 'Diagnosis', 'Excluded', 'ImageCount',
             'Age', 'Sex', 'Height', 'Weight'])

    def __iter__(self):
        for item in self.per_dicom:
            patient_key = item[self.patient_key_name]

            if patient_key in self.returned_patient:
                continue
            else:
                self.returned_patient.add(patient_key)
                yield self.filter(item)

    def filter(self, patient_item):
        new_patient_item = dict()
        for key in patient_item.keys():
            if key in self.pass_patient_keys:
                new_patient_item[key] = patient_item[key]
        return new_patient_item


# 건별 제너레이터(dicom 리스트 반환 버전)
class Per_patient():
    patient_keys_dict = dict()

    def __init__(self, init_data=None, include_excluded=False, search={}):
        self.include_excluded = include_excluded
        self.search = search

        self.patient_key_name = 'AccNo'
        self.pass_patient_keys = set(
            ['AccNo', 'RecoredPatientID', 'RealPatientID',
             'Date', 'Diagnosis', 'Excluded', 'ImageCount',
             'Age', 'Sex', 'Height', 'Weight'])

        # read csv phase only onece
        if init_data:
            self.data = init_data
        else:
            global DEFAULT_INIT_DATA
            self.data = DEFAULT_INIT_DATA

        self.DIAGNOSIS_Dict = self.data.DIAGNOSIS_Dict
        self.DICOM_Dict = self.data.DICOM_Dict
        self.PATIENT_Dict = self.data.PATIENT_Dict

        if not Per_patient.patient_keys_dict:
            per_dicom = Per_dicom(init_data=init_data, include_excluded=True, search={})

            for item in per_dicom:
                patient_key = item[self.patient_key_name]
                if patient_key not in Per_patient.patient_keys_dict:
                    Per_patient.patient_keys_dict[patient_key] = list()
                Per_patient.patient_keys_dict[patient_key].append([item['File'], item['Name'], item['AccNo']])

    def __iter__(self):

        for key in Per_patient.patient_keys_dict.keys():
            l_per_dicom = list()

            for key_diagnosis, key_dicom, key_patient in Per_patient.patient_keys_dict[key]:
                d = dict()
                # merge dict element
                d.update(read_dict(self.DIAGNOSIS_Dict, key_diagnosis))
                d.update(read_dict(self.DICOM_Dict, key_dicom))
                d.update(read_dict(self.PATIENT_Dict, key_patient))

                if self.include_excluded == False:
                    if d['Excluded'] != '':
                        continue

                # match search
                match = True
                for search_key, search_value in self.search.items():
                    if search_key in d.keys():
                        if d[search_key].lower() != search_value.lower():
                            match = False
                            break
                    else:
                        print('not found search key', search_key)
                        raise KeyError

                # return or next
                if match:
                    l_per_dicom.append(d)
                else:
                    continue

            if len(l_per_dicom) > 0:
                d_per_patient = self.filter(l_per_dicom[0])
                yield d_per_patient, l_per_dicom
            else:
                continue

    def filter(self, patient_item):
        new_patient_item = dict()
        for key in patient_item.keys():
            if key in self.pass_patient_keys:
                new_patient_item[key] = patient_item[key]
        return new_patient_item


# 건별 제너레이터(dicom 리스트 반환 버전)
class Per_patient_fast():
    patient_keys_dict = dict()

    def __init__(self, init_data=None, include_excluded=False, search={}):
        self.include_excluded = include_excluded
        self.search = search

        self.patient_key_name = 'AccNo'
        self.pass_patient_keys = set(
            ['AccNo', 'RecoredPatientID', 'RealPatientID',
             'Date', 'Diagnosis', 'Excluded', 'ImageCount',
             'Age', 'Sex', 'Height', 'Weight'])

        # read csv phase only onece
        if init_data:
            self.data = init_data
        else:
            global DEFAULT_INIT_DATA
            self.data = DEFAULT_INIT_DATA

        self.DIAGNOSIS_Dict = self.data.DIAGNOSIS_Dict
        self.DICOM_Dict = self.data.DICOM_Dict
        self.PATIENT_Dict = self.data.PATIENT_Dict

        if not Per_patient.patient_keys_dict:
            per_dicom = Per_dicom(init_data=init_data, include_excluded=True, search={})

            for item in per_dicom:
                patient_key = item[self.patient_key_name]
                if patient_key not in Per_patient.patient_keys_dict:
                    Per_patient.patient_keys_dict[patient_key] = list()
                Per_patient.patient_keys_dict[patient_key].append([item['File'], item['Name'], item['AccNo']])

    def gen_patient(self, key):
        for key_diagnosis, key_dicom, key_patient in Per_patient.patient_keys_dict[key]:
            d = dict()
            # merge dict element
            d.update(read_dict(self.DIAGNOSIS_Dict, key_diagnosis))
            d.update(read_dict(self.DICOM_Dict, key_dicom))
            d.update(read_dict(self.PATIENT_Dict, key_patient))

            if self.include_excluded == False:
                if d['Excluded'] != '':
                    continue

            # match search
            match = True
            for search_key, search_value in self.search.items():
                if search_key in d.keys():
                    if d[search_key].lower() != search_value.lower():
                        match = False
                        break
                else:
                    print('not found search key', search_key)
                    raise KeyError

            # return or next
            if match:
                yield d
            else:
                continue

    def __iter__(self):
        for key in list(Per_patient.patient_keys_dict.keys()):
            try:
                main = self.filter(next(self.gen_patient(key)))
            except StopIteration:
                continue

            sub = self.gen_patient(key)
            yield main, sub

    def filter(self, patient_item):
        new_patient_item = dict()
        for key in patient_item.keys():
            if key in self.pass_patient_keys:
                new_patient_item[key] = patient_item[key]
        return new_patient_item


if __name__ == '__main__':
    pass
else:
    DEFAULT_INIT_DATA = Csv_data()
