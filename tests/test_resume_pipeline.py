"""Offline privacy, fact-selection, escaping and cache checks for Jane Doe."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError
import resume_pipeline as r


def content():
    return {'summary':'Data analyst with a background in Python, SQL and clear analytical reporting.',
            'skills':['Python','SQL','Power BI'],'experience_fact_ids':['E1a','E1b'],
            'project_ids':['P2'],'motivation':'Your emphasis on data quality and practical reporting aligns with my interests.',
            'contribution':'I would apply my SQL and Python background to support reliable analytical workflows.',
            'job_priorities':['Data quality','Reporting'],'gaps':['Additional platform experience is not documented.']}

def job():return {'job_key':'test-job','title':'Data Analyst','company':'Example Company','description':'SQL and Python reporting'}
def profile():return {'model':'test-model','digest':'weights','capabilities':['completion']}


class ResumeTests(unittest.TestCase):
    def test_unknown_skill_is_rejected(self):
        c=content();c['skills'].append('Unlisted certification')
        with self.assertRaises(ValidationError):r.validate_content(c)

    def test_unknown_fact_is_rejected(self):
        c=content();c['experience_fact_ids'].append('invented employer')
        with self.assertRaises(ValidationError):r.validate_content(c)

    def test_numerical_claims_in_free_prose_are_rejected(self):
        c=content();c['summary']='Analyst with 10 years of experience and a proven background.'
        with self.assertRaisesRegex(ValueError,'Unsupported'):r.validate_content(c)

    def test_template_uses_fixed_fictional_identity_and_exact_facts(self):
        docs=r.application_sections(job(),content())
        for section in docs.values():
            self.assertIn('Jane Doe',section);self.assertIn('jane.doe@example.com',section)
            self.assertIn('Fictional Jane Doe',section)
            self.assertIn(r.PROFILE['experience'][0]['facts']['E1a'],section)
        self.assertIn('Example University (fictional)',docs['resume'])

    def test_job_and_generated_text_are_escaped(self):
        j=job();j['title']='<script>alert("x")</script>'
        for section in r.application_sections(j,content()).values():
            self.assertNotIn('<script>',section);self.assertIn('&lt;script&gt;',section)

    def test_cover_letter_voice_is_first_person(self):
        self.assertEqual(r.letter_voice("Jane’s experience will enable her to contribute."),"My experience will enable me to contribute.")
        self.assertEqual(r.letter_voice("Jane will apply her skills."),"I will apply my skills.")

    def test_benefits_are_not_reported_as_qualification_gaps(self):
        c=content();c['gaps']=['401(k) with company matching','Quarterly profit share','Experience with Snowflake']
        self.assertEqual(r.reviewed_gaps(c),['Experience with Snowflake'])

    def test_model_output_is_reused_without_a_second_call(self):
        calls=[]
        def request(*args):
            calls.append(1)
            return {'message':{'content':json.dumps(content())},'done_reason':'stop','prompt_eval_count':40,'eval_count':90}
        with tempfile.TemporaryDirectory() as folder,patch.object(r,'ROOT',Path(folder)):
            first,hit=r.generate_one(job(),profile(),request)
            second,hit2=r.generate_one(job(),profile(),request)
        self.assertFalse(hit);self.assertTrue(hit2);self.assertEqual(len(calls),1)
        self.assertEqual(first['content'],second['content'])

    def test_changed_weights_or_job_invalidate_cache(self):
        old=profile();new=profile();new['digest']='new-weights'
        self.assertNotEqual(r.identity(job(),old),r.identity(job(),new))
        changed=job();changed['description']='Different posting'
        self.assertNotEqual(r.identity(job(),old),r.identity(changed,old))

    def test_truncated_generation_cannot_become_a_final_application(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(r,'ROOT',Path(folder)):
            record,_=r.generate_one(job(),profile(),lambda *args:{'message':{'content':json.dumps(content())},'done_reason':'length'})
        self.assertEqual(record['status'],'invalid')

if __name__=='__main__':unittest.main()
