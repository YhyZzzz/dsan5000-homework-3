"""Check EDA denominators, filtering and identity safeguards offline."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import job_eda as e


def fixture():
    rows=[]
    for index in range(3):
        row={'job_key':f'job-{index}','job_title':'Analyst','company_name':'Example','description_sha256':f'hash-{index}',
             'salary_annual_min':100000,'salary_annual_max':150000,'salary_annual_mid':125000,
             'salary_currency':'USD','salary_issue':None,'experience_years_min':3,'description_word_count':500,
             'work_arrangement':None,'job_type':'Full-time','country':'United States','state':'CA',
             'visa_sponsorship':None,'industry':None,'experience_level':None,
             'crawl_time_utc':'2026-10-30T21:00:00Z','posted_date':'2026-10-29'}
        for field in e.LIST_FIELDS:row[field]=json.dumps(['Python','Python','SQL'] if field=='skills' else [])
        for field in e.BOOL_FIELDS:row[field]=False
        rows.append(row)
    return pd.DataFrame(rows)


class EDATests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def load(self,frame,name='jobs'):
        path=self.root/f'{name}.csv';frame.to_csv(path,index=False)
        return e.JobEDA(path,name,e.COLORS['manual'],self.root/'reports')

    def test_duplicate_identity_is_rejected(self):
        d=fixture();d.loc[1,'job_key']=d.loc[0,'job_key']
        with self.assertRaisesRegex(ValueError,'unique'):self.load(d)

    def test_malformed_list_is_rejected(self):
        d=fixture();d.loc[0,'skills']='{"Python": true}'
        with self.assertRaisesRegex(ValueError,'JSON string array'):self.load(d)

    def test_skill_mentions_are_counted_once_per_posting(self):
        x=self.load(fixture())
        self.assertEqual(e.skill_counts(x.frame)['Python'],3)
        self.assertTrue(x.frame.shared_skill_count.eq(2).all())

    def test_unknown_work_is_retained_in_denominator(self):
        x=self.load(fixture())
        self.assertEqual(e.ordered_counts(x.frame.work_display,e.WORK)['Unknown'],3)
        row=x.coverage().set_index('field').loc['work_arrangement']
        self.assertEqual(row['denominator'],3);self.assertEqual(row['observed'],0)

    def test_salary_exclusions_are_a_union_and_keep_iqr_only_flags(self):
        d=fixture();d.loc[0,'salary_multiple_ranges']=True;d.loc[0,'potential_non_job_posting']=True
        d.loc[1,'salary_currency']='CAD';d['salary_iqr_outlier']=True
        x=self.load(d)
        self.assertEqual(x.salary.job_key.tolist(),['job-2'])
        self.assertFalse(x.frame.loc[2,'potential_non_job_posting'])

    def test_display_normalization_does_not_invent_employment_types(self):
        d=fixture();d.loc[0,'job_type']='Manager';d.loc[1,'job_type']='Contractor'
        x=self.load(d)
        self.assertEqual(x.frame.loc[0,'job_type_display'],'Other/unrecognized')
        self.assertEqual(x.frame.loc[1,'job_type_display'],'Contract')
        self.assertEqual(x.frame.loc[1,'job_type'],'Contractor')

    def test_matching_keeps_only_the_same_cohort(self):
        d=fixture();manual=self.load(d,'manual');sample=self.load(d.iloc[:2],'sample')
        a,b=e.matched_frames(manual,sample)
        self.assertEqual(list(a.index),list(b.index));self.assertEqual(len(a),2)
        self.assertTrue(manual.coverage(a).denominator.eq(2).all())

    def test_unknown_matched_key_is_rejected(self):
        d=fixture();manual=self.load(d,'manual');d.loc[0,'job_key']='unknown';sample=self.load(d,'sample')
        with self.assertRaisesRegex(ValueError,'absent'):e.matched_frames(manual,sample)

    def test_same_key_different_source_is_rejected(self):
        d=fixture();manual=self.load(d,'manual');d.loc[0,'description_sha256']='changed';sample=self.load(d,'sample')
        with self.assertRaisesRegex(ValueError,'Source identity'):e.matched_frames(manual,sample)


if __name__=='__main__':unittest.main()
