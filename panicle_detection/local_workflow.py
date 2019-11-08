'''
Created on Jun 4, 2018

@author: zli
'''
import sys, os, json, argparse, logging, shutil
import panicle_detection
from datetime import date
from globus_sdk import NativeAppAuthClient, TransferClient, TransferData, RefreshTokenAuthorizer

CLIENT_ID = '53669bb5-f588-4c3a-8556-e416ed228a36'


class Transfer:
    def __init__(self, src_endpoint_name, dst_endpoint_name, log_lv=logging.INFO):
        log_format = '%(asctime)-15s %(levelname)s:\t  class:%(name)s %(message)s'
        logging.basicConfig(format=log_format)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_lv)
        self.logger.debug('CLIENT_ID: {0}'.format(CLIENT_ID))
        self.client = NativeAppAuthClient(CLIENT_ID)
        self.client.oauth2_start_flow(refresh_tokens=True)

        authorize_url = self.client.oauth2_get_authorize_url()
        print('Please go to this URL and login: {0}'.format(authorize_url))

        get_input = getattr(__builtins__, 'raw_input', input)
        auth_code = get_input(
            'Please enter the code you get after login here: ').strip()
        token_response = self.client.oauth2_exchange_code_for_tokens(auth_code)
        self.globus_auth_data = token_response.by_resource_server['auth.globus.org']
        self.globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']
        auth_token = self.globus_auth_data['access_token']
        transfer_token = self.globus_transfer_data['access_token']
        transfer_rt = self.globus_transfer_data['refresh_token']
        transfer_at = self.globus_transfer_data['access_token']
        expires_at_s = self.globus_transfer_data['expires_at_seconds']
        self.authorizer = RefreshTokenAuthorizer(transfer_rt, self.client,
                                                 access_token=transfer_at,
                                                 expires_at=expires_at_s)
        self.transferClient = TransferClient(authorizer=self.authorizer)
        self.src_endpoint = None
        self.dst_endpoint = None
        for ep in self.transferClient.endpoint_search(filter_scope="shared-with-me"):
            if ep["display_name"] == src_endpoint_name:
                self.src_endpoint = ep
                self.logger.info('Source endpoint: [{0}] {1}'
                                 .format(self.src_endpoint['id'], self.src_endpoint['display_name']))
        if self.src_endpoint is None:
            self.logger.error('No endpoint shared with you with name: {0}'.format(src_endpoint_name))
            raise LookupError
        for ep in self.transferClient.endpoint_search(filter_scope="my-endpoints"):
            if ep['display_name'] == dst_endpoint_name:
                self.dst_endpoint = ep
                self.logger.info('Destination endpoint: [{0}] {1}'
                                 .format(self.dst_endpoint['id'], self.dst_endpoint['display_name']))
        if self.dst_endpoint is None:
            self.logger.error('You don\'t have endpoint named: {0}'.format(dst_endpoint_name))
            raise LookupError

    def transfer_dir(self, src_dir, dst_dir):
        transfer_data = TransferData(self.transferClient, self.src_endpoint['id'], self.dst_endpoint['id'])
        transfer_data.add_item(src_dir, dst_dir, recursive=True)
        result = self.transferClient.submit_transfer(transfer_data)
        self.logger.info('task [{0}] {1}'.format(result['task_id'], result['code']))
        return result

    def transfer_file(self, src_file, dst_file):
        transfer_data = TransferData(self.transferClient, self.src_endpoint['id'], self.dst_endpoint['id'])
        transfer_data.add_item(src_file, dst_file)
        result = self.transferClient.submit_transfer(transfer_data)
        self.logger.info('task_id [{0}] {1}'.format(result['task_id'], result['code']))
        return result

    def ls_src_dir(self, path, ls_filter=''):
        # using iteration to get every entry from result
        # an entry contain two keys: 'name' and 'type'
        # type define the entry is a file or folder
        result = self.transferClient.operation_ls(self.src_endpoint['id'], path=path, filter=ls_filter)
        for entry in result:
            self.logger.debug('name: {0}\ttype: {1}'.format(entry["name"], entry["type"]))
        return result

    def task_list(self, num_results=10):
        result = self.transferClient.task_list(num_results=num_results)
        for task in result:
            self.logger.debug('task_id: [{0}]\t status: {1}'.format(task['task_id'], task['status']))
        return result

def options():
    
    parser = argparse.ArgumentParser(description='local workflow for Panicle detection from laser data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-m", "--model_path", help="model path")
    parser.add_argument("-p", "--ply_dir", help="ply directory")
    parser.add_argument("-j", "--json_dir", help="json directory")
    parser.add_argument("-o", "--out_dir", help="output directory")
    parser.add_argument("-y", "--year", help="specify which year to process")
    parser.add_argument("-d", "--month", help="specify month")
    parser.add_argument("-s", "--start_date", help="specify start date")
    parser.add_argument("-e", "--end_date", help="specify end date")

    args = parser.parse_args()

    return args

def main():
    
    print("start...")
    
    args = options()
    
    print("create transfer connection")
    t = Transfer('Terraref', 'Zongyang Linux desktop', log_lv=logging.DEBUG)
    tc = t.transferClient
    
    for day in range(int(args.start_date), int(args.end_date)+1):
        target_date = date(int(args.year), int(args.month), day)
        str_date = target_date.isoformat()
        print(str_date)
        
        # globus data transfer
        globus_raw_dir = os.path.join('/ua-mac/raw_data/scanner3DTop/', str_date)
        local_raw_dir = os.path.join('/media/zli/data/Terra/ua-mac/raw_data/scanner3DTop/', str_date)
        
        if not os.path.isdir(local_raw_dir):
            print('start %s raw data transfer'%(str_date))
            task_info = t.transfer_dir(globus_raw_dir, local_raw_dir)
            while not tc.task_wait(task_info['task_id'], timeout=60*5):
                print("Another 5 minute went by without {0} terminating".format(task_info['task_id']))
                
            print('finish raw data transfer')
        
        globus_ply_dir = os.path.join('/ua-mac/Level_1/scanner3DTop/', str_date)
        local_ply_dir = os.path.join('/media/zli/data/Terra/ua-mac/Level_1/scanner3DTop/', str_date)
        
        if not os.path.isdir(local_raw_dir):
            print('start %s ply data transfer'%(str_date))
            task_info = t.transfer_dir(globus_ply_dir, local_ply_dir)
            while not tc.task_wait(task_info['task_id'], timeout=60*5):
                print("Another 5 minute went by without {0} terminating".format(task_info['task_id']))
                
            print('finish ply data transfer')
    
        # start processing
        print('start full day processing...')
        out_dir = os.path.join(args.out_dir, str_date)
        model_file_path = args.model_path
        try:
            panicle_detection.full_day_gen_hist(local_ply_dir, local_raw_dir, out_dir, model_file_path)
                
            if os.path.isdir(out_dir):
                    panicle_detection.full_day_output_integrate(out_dir, out_dir)
                
        except Exception as ex:
            panicle_detection.fail(str_date +str(ex))
        print('finish processing')
            
        # delete globus files
        try:
            shutil.rmtree(local_raw_dir)
            shutil.rmtree(local_ply_dir)
        except Exception as ex:
            panicle_detection.fail(str_date +str(ex))
        
        
    return

if __name__ == "__main__":

    main()