import os
import sys
import argparse

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib


def del_small_file(dir_path, file, file_path, file_size):
    related_file_list = ['-tripinfo.xml', '-queue.xml', '_emission.csv', '-emission.xml', '-ssm.xml']
    if file in os.listdir(dir_path):
        size = os.path.getsize(file_path)
        if size < file_size:
            print()
            print('----------------------')
            print('File size under limit, remove:', file)
            os.remove(file_path)
            file_split = file.split('-')
            file_suffix = ''
            for i in range(0, len(file_split) - 1):
                if i == len(file_split) - 2:
                    file_suffix += file_split[i]
                else:
                    file_suffix += (file_split[i] + '-')
            for each in related_file_list:
                file_name = file_suffix + each
                try:
                    os.remove(os.path.join(dir_path, file_name))
                    print('Remove:', file_name)
                except FileNotFoundError:
                    print('Not exist:', file_name)
            return True
        else:
            return False
    else:
        print()
        print('----------------------')
        print('Already remove:', file)
        return True


def del_not_complete_file(dir_path, file, file_path, horizon):
    related_file_list = ['-tripinfo.xml', '-queue.xml', '_emission.csv', '-ssm.xml']

    if file.endswith('emission.xml'):
        try:
            content = list(sumolib.output.parse(file_path, ['timestep']))
        except:
            recover_xml(file_path)  # unclose tag in xml
            content = list(sumolib.output.parse(file_path, ['timestep']))

        last_timestep = content[-1]
        if not last_timestep.time:  # last: "<timestep/>"
            for i in range(-2, -len(content), -1):
                last_timestep = content[i]
                if float(last_timestep.time):
                    break

        if float(last_timestep.time) < horizon:
            os.remove(file_path)
            print()
            print('----------------------')
            print(last_timestep.time, '< Horizon, remove:', file)
            file_suffix = file.split('-emission.xml')[0]
            for each in related_file_list:
                file_name = file_suffix + each
                try:
                    os.remove(os.path.join(dir_path, file_name))
                    print('Remove:', file_name)
                except FileNotFoundError:
                    print('Not exist:', file_name)


def recover_xml(file_path):
    cmd = "xmllint --valid " + file_path + " --recover --output " + file_path
    print()
    print('-----')
    print('Recover the xml file')
    os.system(cmd)


def recover_all(path, scenario):
    scen_path = os.path.join(path, scenario)
    for eachFile in os.listdir(scen_path):
        if eachFile.endswith('_emission.csv'):
            pass
        else:
            recover_xml(scen_path + "/" + eachFile)


def del_old_file(scen_path, save_dura):
    lastest_time = 0
    for eachFile in os.listdir(scen_path):
        file_info = os.stat(scen_path + "/" + eachFile)
        last_modifed_t = int(file_info.st_mtime)
        lastest_time = (last_modifed_t, lastest_time)[last_modifed_t < lastest_time]
    for eachFile in os.listdir(scen_path):
        file_info = os.stat(scen_path + "/" + eachFile)
        last_modifed_t = int(file_info.st_mtime)
        if lastest_time - last_modifed_t > save_dura:
            os.remove(scen_path + "/" + eachFile)
            print('Remove too old file:', eachFile)


def remove_unfinished_file(path, scenario, horizon, file_size, save_dura, del_old):
    scenarios_list = os.listdir(path)
    if scenario in scenarios_list:
        dir_path = os.path.join(path, scenario)
        if del_old:
            del_old_file(dir_path, save_dura)
        for file in os.listdir(dir_path):
            file_path = dir_path + '/' + file
            if del_small_file(dir_path, file, file_path, file_size):
                pass
            else:
                del_not_complete_file(dir_path, file, file_path, horizon)
        print()
        print('Delete files with incomplete information')
        print('----')
        print()
    else:
        print('Not exist this scenario:', scenario)


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when evaluating an experiment result.",
        epilog="python outputFilesProcessing.py")

    parser.add_argument(
        '--output_dir', type=str, default="../flow/examples/output/",
        help='The directory of output files, default="../flow/examples/output/".')

    parser.add_argument(
        '--scen', type=str,
        help='The scenario name.')

    parser.add_argument(
        '--horizon', type=float, # training horizon: 720, here is recommended about 550, which can filter incomplete output file for SUMO
        help='The time horizon for an episode.')

    parser.add_argument(
        '--file_min', type=int, default=4,
        help='File size limit, default=4.')

    parser.add_argument(
        '--save_duration', type=int, default=300,
        help='Save output files in the last interval, sec, default=300.')

    parser.add_argument(
        '--del_old', type=bool, default=False,
        help='Delete old files, please use once for any scenarios, because the last modified time of emission csv '
             'file does not be updated after using this script, default=False.')

    return parser.parse_known_args(args)[0]


def main(args):
    flags = parse_args(args)
    if flags.scen and flags.horizon:
        path_dir = flags.output_dir
        scenario = flags.scen
        horizon = flags.horizon
        file_size = flags.file_min * 1024
        save_dura = flags.save_duration
        del_old = flags.del_old
        remove_unfinished_file(path_dir, scenario, horizon, file_size, save_dura, del_old)
        recover_all(path_dir, scenario)
    else:
        raise ValueError("Unable to find necessary options: --scen, and --horizon.")


if __name__ == '__main__':
    main(sys.argv[1:])
