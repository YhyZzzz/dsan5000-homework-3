"""Edge-case tests for deterministic parsing and conservative posting deduplication."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import manual_cleaning as m


def record(job_id, title='Data Scientist', company='Example Inc.', location='Boston, MA', description=None):
    return {
        'job': {'job_id': job_id, 'title': title, 'company_name': company, 'location': location,
                'description': description or ('Detailed machine learning position with Python and SQL. ' * 12)},
        'query': 'Data Scientist', 'source_file': 'real-2026-10-30-12-00-00.json',
        'source_index': 0, 'crawl_time': pd.Timestamp('2026-10-30T12:00:00Z'),
    }


class SalaryTests(unittest.TestCase):
    def test_k_range_from_extension(self):
        got = m.parse_salary('120K–180K a year', 'extensions')
        self.assertEqual((got['salary_min'], got['salary_max']), (120000, 180000))
        self.assertEqual(got['salary_annual_mid'], 150000)
        self.assertTrue(got['salary_currency_assumed'])
        self.assertFalse(got['salary_period_assumed'])

    def test_shared_scale_and_inferred_annual_unit_are_explicit(self):
        got = m.parse_salary('Compensation: $65–110k plus benefits')
        self.assertEqual((got['salary_min'],got['salary_max']), (65000,110000))
        self.assertTrue(got['salary_period_assumed'])

    def test_hourly_annualization(self):
        got = m.parse_salary('53.29–85.26 an hour','extensions')
        self.assertAlmostEqual(got['salary_annual_min'],53.29*2080)
        self.assertEqual(got['salary_period'],'hour')

    def test_retirement_plan_is_not_pay(self):
        got = m.parse_salary('Base salary and bonus. 401k with match.\nSalary: $150,000 per year')
        self.assertEqual(got['salary_annual_mid'],150000)
        self.assertEqual(m.parse_salary('Competitive salary, 401K, and paid leave'),{})

    def test_working_hours_do_not_set_salary_period(self):
        got = m.parse_salary('Compensation: $130,000 Work Model: Onsite Hours: 40 hrs a week')
        self.assertEqual(got['salary_period'],'year')
        self.assertTrue(got['salary_period_assumed'])

    def test_generic_hourly_disclaimer_does_not_override_annual_pay(self):
        got = m.parse_salary('$160k - $180k\nThe compensation range or hourly rate listed is an estimate')
        self.assertEqual(got['salary_annual_mid'],170000)

    def test_comma_with_whitespace(self):
        self.assertEqual(m.parse_salary('Starting annual salary is $129, 828')['salary_min'],129828)

    def test_explicit_foreign_currency_is_preserved(self):
        got = m.parse_salary('CAD 100,000–120,000 a year','extensions')
        self.assertEqual(got['salary_currency'],'CAD')
        self.assertFalse(got['salary_currency_assumed'])

    def test_inverted_salary_is_flagged_and_not_annualized(self):
        got = m.parse_salary('$200,000–100,000 per year salary')
        self.assertEqual(got['salary_issue'],'inverted_range')
        self.assertIsNone(got['salary_annual_mid'])

    def test_posting_age_is_not_salary(self):
        self.assertEqual(m.parse_salary('3 days ago','extensions'),{})

    def test_transaction_volume_is_not_salary(self):
        got = m.parse_salary('Payments networks settle more than $2 trillion each day. Salary: $120,000 per year.')
        self.assertEqual(got['salary_annual_mid'],120000)

    def test_separately_labeled_range_endpoints(self):
        got = m.parse_salary('Pay Range - Start:\n$16.50\nPay Range - End:\n$30.00')
        self.assertEqual((got['salary_min'],got['salary_max']),(16.5,30))
        self.assertIsNone(got['salary_annual_mid'])
        self.assertEqual(got['salary_issue'],'period_unknown')


class ExtractionTests(unittest.TestCase):
    def test_posted_date_uses_crawl_timestamp(self):
        result = m.posted_date('4 days ago',pd.Timestamp('2026-10-30T23:00:00Z'))
        self.assertEqual(result[0],pd.Timestamp('2026-10-26T00:00:00Z'))
        self.assertEqual(result[2],'approximately')

    def test_month_and_lower_bound_age(self):
        reference = pd.Timestamp('2026-03-31T12:00:00Z')
        self.assertEqual(m.posted_date('1 month ago',reference)[0],pd.Timestamp('2026-02-28T00:00:00Z'))
        self.assertEqual(m.posted_date('30+ days ago',reference)[2],'on_or_before')

    def test_missing_age_stays_missing(self):
        self.assertTrue(pd.isna(m.posted_date(None,pd.Timestamp('2026-10-30T12:00:00Z'))[0]))

    def test_remote_oregon_is_a_city(self):
        self.assertEqual(m.parse_location('Remote, OR')['city'],'Remote')
        self.assertIsNone(m.work_arrangement('Analyst','Remote, OR',[],'Analyze market risk.')[0])

    def test_multiple_locations_and_us_states(self):
        self.assertEqual(m.parse_location('Denver, CO (+2 others)')['state'],'CO')
        self.assertTrue(m.parse_location('Denver, CO (+2 others)')['multiple_locations'])
        self.assertIsNone(m.parse_location('Anywhere')['country'])

    def test_hybrid_cloud_is_not_a_work_schedule(self):
        self.assertIsNone(m.work_arrangement('Engineer','Boston, MA',[],'Build hybrid cloud infrastructure.')[0])
        self.assertEqual(m.work_arrangement('Engineer','Boston, MA',[],'This is a hybrid work arrangement.')[0],'hybrid')

    def test_skill_boundaries(self):
        got = m.matches('JavaScript, NoSQL and C++ for R&D. Programming in R and Python.',m.SKILLS)
        self.assertIn('C++',got)
        self.assertIn('R',got)
        self.assertNotIn('Java',got)
        self.assertNotIn('SQL',got)
        self.assertNotIn('R',m.matches('Research and R&D only.',m.SKILLS))

    def test_ms_office_and_master_data_are_not_masters_degrees(self):
        self.assertEqual(m.matches('MS Office and master data governance',m.DEGREES),[])
        self.assertEqual(m.matches(m.clean_text('Master’s degree in Statistics'),m.DEGREES),['Master'])

    def test_degree_alternatives_use_lowest_experience_path(self):
        got = m.experience("Bachelor's degree with 8 years of experience OR Master's degree with 6 years of experience.")
        self.assertEqual(got['experience_years_min'],6)

    def test_age_and_preferred_experience_are_excluded(self):
        got = m.experience('At least 18 years of age.\n3+ years of Python experience required.\n10 years of experience preferred.')
        self.assertEqual(got['experience_years_min'],3)

    def test_sponsorship_negation_and_irrelevant_sponsorship(self):
        self.assertEqual(m.sponsorship('We will not provide visa sponsorship for this position.')[0],'not_available')
        self.assertEqual(m.sponsorship('We will consider sponsoring a new applicant for employment authorization.')[0],'conditional')
        self.assertEqual(m.sponsorship('We offer conference sponsorship opportunities.'),(None,None))

    def test_explicit_date_requires_a_year(self):
        self.assertEqual(m.explicit_date('Apply by November 15, 2026','apply by')[0],pd.Timestamp('2026-11-15',tz='UTC'))
        self.assertTrue(pd.isna(m.explicit_date('Apply by November 15','apply by')[0]))


class DeduplicationTests(unittest.TestCase):
    def test_identical_description_across_locations_retains_provenance(self):
        records=[record('a'),record('b',location='Denver, CO')]
        groups,_,_=m.deduplicate(records)
        self.assertEqual(len(groups),1)
        row=m.extract_group(records,*groups[0])
        self.assertEqual(row['source_record_count'],2)
        self.assertEqual(len(row['locations']),2)
        self.assertEqual(row['job_ids'],['a','b'])

    def test_different_companies_are_not_merged_on_boilerplate(self):
        groups,_,_=m.deduplicate([record('a'),record('b',company='Other Inc.')])
        self.assertEqual(len(groups),2)

    def test_canonical_key_with_distinct_description_is_only_flagged(self):
        groups,_,reviews=m.deduplicate([record('a'),record('b',description='A completely different research position. ' * 20)])
        self.assertEqual(len(groups),2)
        self.assertEqual(len(reviews),1)
        self.assertTrue(all(g[2] for g in groups))

    def test_short_generic_description_does_not_merge_distinct_titles(self):
        groups,_,_=m.deduplicate([record('a',description='We are hiring.'),record('b',title='Engineer',description='We are hiring.')])
        self.assertEqual(len(groups),2)

    def test_missing_optional_fields_do_not_crash_extraction(self):
        row=record('a');row['job']={'job_id':'a','title':'Analyst','description':'Analyze data.'}
        result=m.extract_group([row],[0],[],False)
        self.assertIsNone(result['salary_min'])
        self.assertIsNone(result['company_name'])
        self.assertEqual(result['skills'],[])

    def test_example_file_is_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            payload={'search_metadata':{'created_at':'2026-10-30 12:00:00 UTC'},'jobs_results':[record('a')['job']]}
            for name in ['real.json','example-out-old.json']:
                (Path(folder)/name).write_text(json.dumps(payload))
            self.assertEqual(len(m.read_raw(folder)),1)


if __name__=='__main__':
    unittest.main()
