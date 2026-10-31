"""Offline contract tests; fabricated responses never enter production model caches."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

import llm_cleaning as l


def empty_features():
    return {
        'salary':dict(min=None,max=None,period=None,currency=None,quote=None),
        'experience':dict(min_years=None,max_years=None,level=None,quote=None),
        'work':dict(value=None,quote=None),'visa':dict(value=None,quote=None),'industry':dict(value=None,quote=None),
        'education_degrees':[],'skills':[],'benefits':[],'job_types':[],'certifications':[],
        'soft_skills':[],'company_values':[],'collaboration_tools':[],'required_languages':[],
        'company_employees':None,'team_size':None,'travel_percent_max':None,
        'dates':dict(application_deadline=None,expected_start_date=None,posting_expiry_date=None),
    }


def posting():
    return {'job_key':'example-key','payload':{
        'title':'Senior Data Scientist','company_name':'Example','location':'Boston, MA',
        'extensions':['Full-time'],
        'description':'Salary: $100K–$150K per year. 5+ years of Python and SQL experience. Fully remote work. Apply by November 15, 2026.',
    }}


def profile():
    return {'model':'test-model','digest':'test-digest','capabilities':['completion']}


class SchemaTests(unittest.TestCase):
    def test_complete_nullable_object_is_valid(self):
        self.assertEqual(l.JobFeatures.model_validate(empty_features()).skills,[])

    def test_invalid_category_rejected(self):
        f=empty_features();f['work']['value']='Contract'
        with self.assertRaises(ValidationError):l.JobFeatures.model_validate(f)

    def test_extra_field_rejected(self):
        f=empty_features();f['made_up']='x'
        with self.assertRaises(ValidationError):l.JobFeatures.model_validate(f)

    def test_impossible_travel_percentage_rejected(self):
        f=empty_features();f['travel_percent_max']=150
        with self.assertRaises(ValidationError):l.JobFeatures.model_validate(f)


class EvidenceTests(unittest.TestCase):
    def test_quote_must_be_an_actual_substring(self):
        self.assertTrue(l.quote_supported('Fully remote work',posting()['payload']))
        self.assertFalse(l.quote_supported('Hybrid work permitted',posting()['payload']))

    def test_quote_normalizes_thousands_separator_spacing(self):
        self.assertTrue(l.quote_supported('$129,828',{'description':'Annual salary is $129, 828.'}))

    def test_a_quoted_range_endpoint_is_not_an_exact_salary(self):
        f=empty_features();f['salary']=dict(min=150000,max=None,period='year',currency='USD',quote='$150K')
        got,issues=l.normalize_features(f,posting()['payload'])
        self.assertIsNone(got['salary']['min'])
        self.assertIn('salary:incomplete_bounds',issues)

    def test_unsupported_salary_is_not_filled_from_rules(self):
        f=empty_features();f['salary']=dict(min=100000,max=150000,period='year',currency='USD',quote='A competitive salary is provided')
        got,issues=l.normalize_features(f,posting()['payload'])
        self.assertIsNone(got['salary']['min'])
        self.assertIn('salary:unsupported_or_missing_quote',issues)

    def test_numeric_salary_hallucination_rejected(self):
        f=empty_features();f['salary']=dict(min=200000,max=300000,period='year',currency='USD',quote='$100K–$150K per year')
        got,issues=l.normalize_features(f,posting()['payload'])
        self.assertIsNone(got['salary']['min'])
        self.assertIn('salary:numeric_evidence_mismatch',issues)

    def test_swapped_bounds_are_reordered_and_flagged(self):
        f=empty_features();f['salary']=dict(min=150000,max=100000,period='year',currency='USD',quote='$100K–$150K per year')
        got,issues=l.normalize_features(f,posting()['payload'])
        self.assertEqual((got['salary']['min'],got['salary']['max']),(100000,150000))
        self.assertIn('salary:inverted_bounds_reordered',issues)

    def test_shared_k_scale(self):
        self.assertTrue({65000,110000} <= l.quoted_amounts('$65–110K'))

    def test_salary_quote_with_space_after_comma(self):
        self.assertIn(129828,l.quoted_amounts('Starting annual salary is $129, 828'))

    def test_retirement_plan_is_not_salary_evidence(self):
        self.assertNotIn(401000,l.quoted_amounts('401k with employer match'))

    def test_omitted_skills_remain_omitted(self):
        f=empty_features()
        got,_=l.normalize_features(f,posting()['payload'])
        self.assertEqual(got['skills'],[])  # Never silently substitute manual labels.

    def test_unsupported_skill_proposal_is_removed(self):
        f=empty_features();f['skills']=['Python','JavaScript']
        got,issues=l.normalize_features(f,posting()['payload'])
        self.assertEqual(got['skills'],['Python'])
        self.assertIn('skills:unsupported_mention:JavaScript',issues)

    def test_model_invented_date_is_rejected(self):
        f=empty_features();f['dates']['application_deadline']='2026-12-31'
        got,issues=l.normalize_features(f,posting()['payload'])
        self.assertIsNone(got['dates']['application_deadline'])
        self.assertIn('application_deadline:unsupported_date',issues)

    def test_actual_explicit_date_is_kept(self):
        f=empty_features();f['dates']['application_deadline']='2026-11-15'
        got,_=l.normalize_features(f,posting()['payload'])
        self.assertEqual(got['dates']['application_deadline'],'2026-11-15')

    def test_seniority_must_match_title(self):
        f=empty_features();f['experience']['level']='Entry/Junior'
        got,issues=l.normalize_features(f,posting()['payload'])
        self.assertIsNone(got['experience']['level'])
        self.assertIn('experience:unsupported_title_level',issues)

    def test_unstated_language_and_job_type_are_rejected(self):
        f=empty_features();f['required_languages']=['Spanish'];f['job_types']=['Full-time','Internship']
        got,_=l.normalize_features(f,posting()['payload'])
        self.assertEqual(got['job_types'],['Full-time'])
        self.assertEqual(got['required_languages'],[])


class CacheTests(unittest.TestCase):
    def fake_response(self,reason='stop'):
        return {'message':{'content':json.dumps(empty_features())},'done_reason':reason,
                'prompt_eval_count':100,'eval_count':100,'total_duration':1000}

    def test_successful_response_is_reused_without_request(self):
        calls=[]
        def fake(endpoint,payload):calls.append(payload);return self.fake_response()
        with tempfile.TemporaryDirectory() as folder:
            first,hit=l.extract_one(posting(),profile(),folder,fake)
            second,hit2=l.extract_one(posting(),profile(),folder,fake)
        self.assertFalse(hit);self.assertTrue(hit2);self.assertEqual(len(calls),1)
        self.assertEqual(first['features'],second['features'])

    def test_model_digest_changes_cache_identity(self):
        old=profile();new=profile();new['digest']='updated-weights'
        self.assertNotEqual(l.cache_identity(posting(),old),l.cache_identity(posting(),new))

    def test_source_changes_cache_identity(self):
        changed=copy.deepcopy(posting());changed['payload']['description']+=' Updated pay.'
        self.assertNotEqual(l.cache_identity(posting(),profile()),l.cache_identity(changed,profile()))

    def test_truncated_output_is_saved_but_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            r,hit=l.extract_one(posting(),profile(),folder,lambda *args:self.fake_response('length'))
            self.assertEqual(r['status'],'invalid')
            self.assertTrue(list(Path(folder).glob('*.json')))

    def test_failed_attempt_history_is_preserved(self):
        outputs=iter([self.fake_response('length'),self.fake_response()])
        with tempfile.TemporaryDirectory() as folder:
            l.extract_one(posting(),profile(),folder,lambda *args:next(outputs))
            r,_=l.extract_one(posting(),profile(),folder,lambda *args:next(outputs))
        self.assertEqual(r['status'],'validated')
        self.assertEqual(len(r['previous_attempts']),1)

    def test_incomplete_cache_cannot_export_as_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'incomplete'):
                l.cached_dataset([posting()],profile(),folder)


class MatchedSampleTests(unittest.TestCase):
    def make_frames(self):
        import pandas as pd
        fields=['salary_annual_mid','experience_years_min','work_arrangement',
                'visa_sponsorship','job_type','experience_level','industry']
        baseline=pd.DataFrame([{'job_key':key,**{f:None for f in fields}} for key in ['a','b','c']])
        baseline['job_type']='Full-time'
        return baseline,baseline.iloc[:2].copy()

    def test_sample_comparison_uses_only_the_same_keys(self):
        baseline,candidate=self.make_frames()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manual.csv';baseline.to_csv(path,index=False)
            summary,_=l.comparison_tables(candidate,path,allow_subset=True)
        row=summary.set_index('field').loc['job_type']
        self.assertEqual(row['manual_observed'],2)
        self.assertEqual(row['llm_observed'],2)
        self.assertEqual(row['both_agree'],2)

    def test_subset_must_be_explicit(self):
        baseline,candidate=self.make_frames()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manual.csv';baseline.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'same posting'):
                l.comparison_tables(candidate,path)

    def test_unknown_sample_key_is_never_accepted(self):
        baseline,candidate=self.make_frames();candidate.loc[0,'job_key']='unknown'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manual.csv';baseline.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'unknown posting'):
                l.comparison_tables(candidate,path,allow_subset=True)


if __name__=='__main__':unittest.main()
