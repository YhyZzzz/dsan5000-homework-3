"""Deterministic cleaning for HW 3.2. No network calls or LLM runtime dependencies."""

from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


STATES = dict(zip(
    'AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY'.split(),
    'Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|District of Columbia|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming'.split('|')
))
SKILLS = {
    'Python': r'\bpython\b', 'R': r'(?<![\w])R(?!\w|\s*&\s*D)',
    'SQL': r'\bsql\b', 'NoSQL': r'\bnosql\b', 'Java': r'\bjava\b',
    'JavaScript': r'\bjavascript\b', 'C++': r'(?<!\w)c\+\+(?!\w)',
    'C#': r'(?<!\w)c#(?!\w)', 'Scala': r'\bscala\b', 'SAS': r'\bsas\b',
    'MATLAB': r'\bmatlab\b', 'Excel': r'\bexcel\b', 'Tableau': r'\btableau\b',
    'Power BI': r'\bpower\s*bi\b', 'Spark': r'\b(?:apache\s+|py)?spark\b',
    'Hadoop': r'\bhadoop\b', 'Kafka': r'\bkafka\b', 'Airflow': r'\bairflow\b',
    'dbt': r'\bdbt\b', 'Snowflake': r'\bsnowflake\b', 'Databricks': r'\bdatabricks\b',
    'AWS': r'\baws\b|\bamazon web services\b', 'Azure': r'\bazure\b',
    'GCP': r'\bgcp\b|\bgoogle cloud\b', 'Docker': r'\bdocker\b',
    'Kubernetes': r'\bkubernetes\b|\bk8s\b', 'Git': r'\bgit\b|\bgithub\b',
    'TensorFlow': r'\btensorflow\b', 'PyTorch': r'\bpytorch\b',
    'scikit-learn': r'\bscikit[ -]learn\b|\bsklearn\b',
    'Machine learning': r'\bmachine learning\b|\bML\b',
    'Deep learning': r'\bdeep learning\b', 'NLP': r'\bnlp\b|\bnatural language processing\b',
    'Computer vision': r'\bcomputer vision\b',
    'LLM': r'\bllms?\b|\blarge language models?\b',
    'RAG': r'\brag\b|\bretrieval[ -]augmented generation\b',
    'LangChain': r'\blangchain\b', 'LangGraph': r'\blanggraph\b',
    'Statistics': r'\bstatistics\b|\bstatistical\b',
    'A/B testing': r'\ba/b test\w*|\bab testing\b',
    'Optimization': r'\boptimization\b|\boptimisation\b',
    'MLOps': r'\bmlops\b', 'Linux': r'\blinux\b',
}
DEGREES = {
    'High school': r'\bhigh school\b|\bged\b',
    'Associate': r"\bassociate(?:'s)? degree\b",
    'Bachelor': r"\bbachelor(?:'s|s)?\b|\bB\.S\.(?!\w)|\bB\.A\.(?!\w)|\b(?:BS|BA)\s+(?:degree|in\b)",
    'Master': r"\bmaster(?:'s|s)\b|\bmaster (?:degree|of)\b|\bM\.S\.(?!\w)|\bM\.A\.(?!\w)|\b(?:MS|MA)\s+(?:degree|in\b)|\bMBA\b",
    'Doctorate': r'\bph\.?d\.?\b|\bdoctorate\b|\bdoctoral\b',
}
CERTIFICATIONS = {
    'AWS Certified': r'\baws certif\w*', 'PMP': r'\bpmp\b', 'CISSP': r'\bcissp\b',
    'CFA': r'\bcfa\b', 'CPA': r'\bcpa\b', 'Security+': r'\bsecurity\+',
    'Azure certification': r'\bazure.{0,30}certif\w*',
    'Google Cloud certification': r'\bgoogle cloud.{0,30}certif\w*',
    'CompTIA': r'\bcomptia\b', 'Six Sigma': r'\bsix sigma\b',
}
BENEFITS = {
    'Health insurance': r'\bhealth (?:insurance|coverage)\b|\bmedical (?:insurance|coverage)\b',
    'Dental insurance': r'\bdental\b', 'Vision insurance': r'\bvision (?:insurance|coverage|benefits)\b',
    'Paid time off': r'\bpaid time off\b|\bpto\b|\bpaid (?:vacation|leave|holidays)\b',
    'Retirement': r'\b401\s*\(?k\)?|\bretirement (?:plan|benefit|saving)\w*|\bpension\b',
    'Stock/equity': r'\bstock options?\b|\brestricted stock\b|\bequity (?:award|grant|compensation)\w*|\brsus?\b',
    'Bonus': r'\bbonus(?:es)?\b', 'Learning support': r'\btuition (?:reimbursement|assistance)\b|\bprofessional development\b|\blearning (?:budget|stipend)\b',
    'Parental leave': r'\bparental leave\b|\bmaternity leave\b|\bpaternity leave\b',
}
SOFT_SKILLS = {name: pattern for name, pattern in [
    ('Communication', r'\bcommunication\b|\bcommunicat\w*'),
    ('Leadership', r'\bleadership\b'), ('Collaboration', r'\bcollaborat\w*|\bteamwork\b'),
    ('Problem solving', r'\bproblem[ -]solving\b'), ('Critical thinking', r'\bcritical thinking\b'),
    ('Presentation', r'\bpresentation\w*'), ('Stakeholder management', r'\bstakeholder\w*'),
]}
CULTURE = {'Diversity/inclusion': r'\bdiversity\b|\binclusion\b|\binclusive\b',
           'Innovation': r'\binnovation\b|\binnovative\b',
           'Sustainability': r'\bsustainab\w*', 'Customer focus': r'\bcustomer[ -]centric\b|\bcustomer focus\b'}
TOOLS = {name: rf'\b{pattern}\b' for name, pattern in [
    ('Slack', 'slack'), ('Zoom', 'zoom'), ('Microsoft Teams', 'microsoft teams'),
    ('Jira', 'jira'), ('Confluence', 'confluence')
]}
LIST_COLUMNS = [
    'job_ids', 'alternate_titles', 'locations', 'posting_platforms', 'job_types',
    'search_queries', 'source_files', 'qualifications', 'responsibilities',
    'education_degrees', 'skills', 'certifications', 'soft_skills', 'benefits',
    'company_values', 'collaboration_tools', 'required_languages', 'duplicate_match_rules',
]
NUMERIC_COLUMNS = [
    'salary_min', 'salary_max', 'salary_annual_min', 'salary_annual_max', 'salary_annual_mid',
    'experience_years_min', 'experience_years_max', 'company_employees', 'team_size', 'travel_percent_max',
]
DATE_COLUMNS = ['posted_date', 'application_deadline', 'expected_start_date', 'posting_expiry_date']
INTEGER_COLUMNS = ['source_record_count', 'description_word_count', 'description_char_count', 'skill_count']


def clean_text(value):
    if value is None:
        return ''
    return re.sub(r'[ \t\r\f\v]+', ' ', unicodedata.normalize('NFKC', html.unescape(str(value))).translate(str.maketrans({"’": "\'", "‘": "\'", "\u200b": ""}))).strip()


def normalized(value):
    return re.sub(r'[^\w]+', ' ', clean_text(value).casefold()).strip()


def matches(text, vocabulary):
    # R is case-sensitive: lower-case articles/letters are not the R language.
    return sorted(name for name, pattern in vocabulary.items()
                  if re.search(pattern, text, 0 if name == 'R' else re.I))


def clauses(text):
    return [s.strip() for s in re.split(r'\n|;|(?<=[.!?])\s+(?=[A-Z])', text) if s.strip()]


def read_raw(data_dir):
    """Keep every raw occurrence and provenance; deliberately exclude the old example."""
    records = []
    for path in sorted(Path(data_dir).glob('*.json')):
        if path.name.startswith('example-out-'):
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        if 'jobs_results' not in payload:
            continue
        metadata = payload.get('search_metadata', {})
        crawl = payload.get('crawl_metadata', {})
        stamp = metadata.get('created_at') or crawl.get('collected_at_utc')
        if not stamp:
            match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', path.name)
            stamp = pd.to_datetime(match[1], format='%Y-%m-%d-%H-%M-%S', utc=True) if match else None
        when = pd.to_datetime(str(stamp).replace(' UTC', '+00:00'), utc=True, errors='coerce')
        if pd.isna(when):
            raise ValueError(f'Missing usable crawl timestamp: {path.name}')
        query = crawl.get('query') or payload.get('search_parameters', {}).get('q')
        for index, job in enumerate(payload['jobs_results']):
            if not isinstance(job, dict):
                raise ValueError(f'Non-dictionary job in {path.name}')
            records.append({'job': job, 'query': query, 'source_file': path.name,
                            'crawl_time': when, 'source_index': index})
    if not records:
        raise ValueError('No real crawl records were found.')
    return records


def deduplicate(records, threshold=0.95):
    """Link exact IDs, company+identical descriptions, and very similar canonical matches.

    Identical descriptions of at least 40 words at several locations become one
    posting with all locations preserved. Matching title/company/location alone
    is insufficient for removal; short boilerplate cannot merge distinct titles.
    """
    parents = list(range(len(records)))
    reasons = defaultdict(set)
    edges, reviews = [], []

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def link(a, b, reason, similarity=1.0):
        parents[find(b)] = find(a)
        edges.append({'left_raw_index': a, 'right_raw_index': b,
                      'reason': reason, 'description_similarity': similarity})

    ids, descriptions, canonical = {}, {}, defaultdict(list)
    for i, record in enumerate(records):
        job = record['job']
        company = normalized(job.get('company_name'))
        title = normalized(job.get('title'))
        location = normalized(job.get('location'))
        description = normalized(job.get('description'))
        job_id = clean_text(job.get('job_id'))
        if job_id:
            if job_id in ids:
                link(ids[job_id], i, 'same_job_id')
            ids[job_id] = i
        if company and len(description.split()) >= 40:
            key = (company, hashlib.sha256(description.encode()).hexdigest())
            if key in descriptions:
                link(descriptions[key], i, 'same_company_and_description')
            descriptions[key] = i
        if company and title and location:
            key = (company, title, location)
            for other in canonical[key]:
                if find(i) == find(other):
                    continue
                previous = normalized(records[other]['job'].get('description'))
                similarity = SequenceMatcher(None, previous.split(), description.split(), autojunk=False).ratio()
                if description and previous and similarity >= threshold:
                    link(other, i, 'canonical_key_and_similar_description', similarity)
                else:
                    reviews.append({'left_raw_index': other, 'right_raw_index': i,
                                    'reason': 'same_title_company_location_but_distinct_description',
                                    'description_similarity': similarity})
            canonical[key].append(i)
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    for edge in edges:
        reasons[find(edge['left_raw_index'])].add(edge['reason'])
    review_indices = {i for item in reviews for i in [item['left_raw_index'], item['right_raw_index']]}
    return [(indices, sorted(reasons[find(indices[0])]), bool(set(indices) & review_indices))
            for indices in groups.values()], edges, reviews


def parse_location(value):
    raw = clean_text(value)
    primary = re.sub(r'\s*\(\+\d+ others?\)', '', raw).strip()
    result = {'city': None, 'state': None, 'country': None,
              'multiple_locations': bool(re.search(r'\(\+\d+ others?\)', raw))}
    if primary.casefold() in {'anywhere', 'remote', ''}:
        return result
    if primary.casefold() in {'united states', 'usa', 'us', 'u.s.'}:
        result['country'] = 'United States'
    elif primary in STATES.values():
        result.update(state=next(code for code, name in STATES.items() if name == primary), country='United States')
    else:
        found = re.fullmatch(r'(.+),\s*([A-Z]{2})(?:,?\s*(?:USA|United States))?', primary)
        if found and found[2] in STATES:
            result.update(city=found[1], state=found[2], country='United States')
    return result


NUMBER = r'(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)'
CURRENCY = r'(?:US\$|USD|CAD|C\$|GBP|EUR|\$|£|€)'
MONEY = re.compile(
    rf'(?<![\w.])(?P<c1>{CURRENCY})?\s*(?P<n1>{NUMBER})\s*(?P<k1>[kK])?'
    rf'(?:\s*(?:[-–—]|\bto\b)\s*(?P<c2>{CURRENCY})?\s*(?P<n2>{NUMBER})\s*(?P<k2>[kK])?)?', re.I)
PERIOD = re.compile(r'(?i)^\s*(?:(?:p/|per\s+|an?\s+|/)\s*(hour|hr|year|yr|annum|month|week|day)s?\b|/?\s*(hourly|annually|annual|yearly|monthly|weekly|daily)\b)')


def parse_salary(text, source='description'):
    """Prefer explicit periods; annual inference is retained and flagged separately."""
    text = re.sub(r'(?<=\d),\s+(?=\d{3}\b)', ',', clean_text(text))
    # Some postings label the endpoints on separate lines instead of using a dash.
    text = re.sub(
        rf'(?i)pay range\s*-\s*start\s*:\s*({CURRENCY}?\s*{NUMBER}[kK]?)\s*pay range\s*-\s*end\s*:\s*({CURRENCY}?\s*{NUMBER}[kK]?)',
        r'Pay range: \1 - \2', text,
    )
    for match in MONEY.finditer(text):
        context = text[max(0, match.start()-85): min(len(text), match.end()+85)]
        after = text[match.end():match.end()+35]
        if re.match(r'(?i)\s*(?:million|billion|trillion)\b', after):
            continue  # Company transaction volumes and market size are not pay.
        unit = PERIOD.match(after)
        prefix_unit = re.search(r'(?i)\b(hourly|annual|yearly|monthly|weekly|daily)\s+(?:base\s+)?(?:salary|pay|wage|compensation)(?:\s+\w+){0,4}\s*[:=]?\s*$', text[max(0,match.start()-70):match.start()])
        if match['n1'] == '401' and match['k1'] and not match['c1'] and not unit:
            continue
        marker = match['c1'] or match['c2']
        has_scale = bool(match['k1'] or match['k2'])
        if source not in {'extensions', 'detected_extensions'}:
            if not (marker or has_scale) or not re.search(r'(?i)salary|compensation|pay\b|wage|base', context):
                continue
            # Numbers earlier in a compensation sentence are not necessarily pay.
            if not marker and not has_scale:
                continue
        elif not (marker or has_scale or (unit and PERIOD.search(after))):
            continue
        if re.match(r'\s*%', text[match.end():]):
            continue
        scale1 = 1000 if match['k1'] else 1
        scale2 = 1000 if match['k2'] else 1
        if match['n2'] and match['k1'] != match['k2'] and has_scale:
            scale1 = scale2 = 1000
        low = float(match['n1'].replace(',', '')) * scale1
        high = float((match['n2'] or match['n1']).replace(',', '')) * (scale2 if match['n2'] else scale1)
        period = None
        if unit or prefix_unit:
            name = ((unit[1] or unit[2]) if unit else prefix_unit[1]).lower()
            period = next(label for label, values in {
                'hour': ['hour', 'hr', 'hourly'], 'year': ['year', 'yr', 'annum', 'annual', 'annually', 'yearly'],
                'month': ['month', 'monthly'], 'week': ['week', 'weekly'], 'day': ['day', 'daily'],
            }.items() if name in values)
        inferred_period = period is None and low >= 10000 and high >= 10000
        if inferred_period:
            period = 'year'
        currency = None
        if marker:
            currency = 'CAD' if marker.upper() in {'CAD', 'C$'} else 'GBP' if marker.upper() in {'GBP', '£'} else 'EUR' if marker.upper() in {'EUR', '€'} else 'USD'
        assumed_currency = currency is None
        currency = currency or 'USD'  # U.S.-targeted search; flagged, not silently treated as explicit.
        issue = 'inverted_range' if low > high else 'nonpositive_amount' if low <= 0 else None
        if period is None and issue is None:
            issue = 'period_unknown'
        factor = {'year': 1, 'month': 12, 'week': 52, 'day': 260, 'hour': 2080}.get(period)
        annual_low = low * factor if factor and not issue else None
        annual_high = high * factor if factor and not issue else None
        return {
            'salary_min': low, 'salary_max': high, 'salary_period': period,
            'salary_currency': currency, 'salary_currency_assumed': assumed_currency,
            'salary_period_assumed': inferred_period, 'salary_annual_min': annual_low,
            'salary_annual_max': annual_high,
            'salary_annual_mid': (annual_low + annual_high) / 2 if annual_low is not None else None,
            'salary_evidence': text[max(0,match.start()-55):min(len(text),match.end()+45)],
            'salary_source': source, 'salary_issue': issue,
            'salary_multiple_ranges': sum(bool(m['n2'] and (m['c1'] or m['c2'] or m['k1'] or m['k2'])) for m in MONEY.finditer(text)) > 1,
        }
    return {}


def posted_date(label, crawl_time):
    """Relative dates are estimates anchored to the response timestamp, never today."""
    label = clean_text(label).casefold()
    if label in {'today', 'just posted', 'just now'}:
        return crawl_time.normalize(), 'relative_day', 'approximately'
    if label == 'yesterday':
        return (crawl_time-pd.Timedelta(days=1)).normalize(), 'relative_day', 'approximately'
    found = re.fullmatch(r'(\d+)\s*(\+)?\s*(minute|hour|day|week|month|year)s? ago', label)
    if not found:
        return pd.NaT, None, None
    count, plus, unit = int(found[1]), found[2], found[3]
    if unit in {'month', 'year'}:
        stamp = crawl_time - pd.DateOffset(**{unit+'s': count})
    else:
        stamp = crawl_time - pd.Timedelta(**{unit+'s': count})
    return stamp.normalize(), 'relative_'+unit, 'on_or_before' if plus else 'approximately'


def experience(text):
    candidates = []
    for sentence in clauses(text):
        if not re.search(r'(?i)experience|background|working|worked', sentence):
            continue
        if re.search(r'(?i)preferred|nice.to.have|a plus|bonus|desirable', sentence):
            continue
        sentence_candidates = []
        for found in re.finditer(r'(?i)\b(\d{1,2})(?:\s*[-–—]\s*(\d{1,2}))?\s*\+?\s*years?\b', sentence):
            after = sentence[found.end():found.end()+30]
            if re.search(r'(?i)^\s*(?:of\s+)?(?:age|degree|college|university)', after):
                continue
            before = sentence[max(0,found.start()-40):found.start()]
            local = before + sentence[found.start():found.end()+75]
            if not re.search(r'(?i)experience|background|working|worked|overall', local):
                continue
            sentence_candidates.append((int(found[1]), int(found[2]) if found[2] else None, sentence[:700]))
        if sentence_candidates:
            degree_alternatives = re.search(r'(?i)\bor\s+(?:an?\s+)?(?:master|ph\.?d|doctor|bachelor|associate)', sentence)
            choose = min if degree_alternatives else max
            candidates.append(choose(sentence_candidates, key=lambda item:item[0]))
    if not candidates:
        return {'experience_years_min': None, 'experience_years_max': None, 'experience_evidence': None}
    low, high, evidence = max(candidates, key=lambda item: item[0])
    return {'experience_years_min': low, 'experience_years_max': high, 'experience_evidence': evidence}


def seniority(title):
    for label, pattern in [
        ('Executive', r'\bchief\b|\bvp\b|vice president'), ('Director', r'\bdirector\b'),
        ('Manager', r'\bmanager\b'), ('Principal/Staff/Lead', r'\bprincipal\b|\bstaff\b|\blead\b'),
        ('Senior', r'\bsenior\b|\bsr\.?\b'), ('Intern', r'\bintern(?:ship)?\b'),
        ('Entry/Junior', r'\bjunior\b|\bjr\.?\b|entry[ -]level|new grad'),
    ]:
        if re.search(pattern, title, re.I):
            return label
    return None


def work_arrangement(title, location, extensions, text):
    short = title + '\n' + '\n'.join(extensions)
    remote = location.casefold() == 'anywhere' or bool(re.search(r'(?i)\bwork from home\b|\bremote\b(?!\s+sensing)', short))
    onsite, hybrid, evidence = False, False, []
    for sentence in clauses(text):
        if re.search(r'(?i)\bhybrid\s+(?:work|schedule|role|position|arrangement)|(?:work arrangement|work model|work schedule)\s*:?\s*hybrid', sentence):
            hybrid = True; evidence.append(sentence[:400])
        if re.search(r'(?i)\b(?:fully |100% )?remote (?:role|position|work|opportunity)|\bwork(?:ing)? remotely\b|\bwork from home\b|\b(?:role|position) is remote\b', sentence):
            if not re.search(r'(?i)\bnot\b|\bno remote\b|not available', sentence):
                remote = True; evidence.append(sentence[:400])
        if re.search(r'(?i)\bon[ -]?site\b|\bonsite\b', sentence) and re.search(r'(?i)work|role|position|office|expect|designated|designed', sentence):
            if not re.search(r'(?i)\bnot\b|\bno on[ -]?site\b', sentence):
                onsite = True; evidence.append(sentence[:400])
    if re.search(r'(?i)\bhybrid\b', title):
        hybrid = True
    category = 'hybrid' if hybrid else 'conflicting' if onsite and remote else 'remote' if remote else 'onsite' if onsite else None
    return category, '\n'.join(dict.fromkeys(evidence)) or (short[:400] if remote or hybrid else None)


def sponsorship(text):
    evidence = [s for s in clauses(text) if re.search(r'(?i)sponsor', s) and re.search(r'(?i)visa|immigra|h-?1b|employment|work authori[sz]|sponsor(?:ship)? (?:is |will |not |available)', s)]
    if not evidence:
        return None, None
    joined = ' '.join(evidence)
    negative = bool(re.search(r'(?i)\b(?:no|without).{0,45}sponsor|(?:will not|cannot|unable|not eligible|does not|not provide).{0,100}sponsor|sponsor.{0,35}(?:not available|not offered)', joined))
    conditional = bool(re.search(r'(?i)consider sponsor|may (?:offer|provide|be available)|h-?1b transfer', joined))
    positive = bool(re.search(r'(?i)(?:will|can) (?:provide )?sponsor|sponsorship (?:is )?(?:available|offered)|offer.{0,30}visa sponsorship', joined))
    status = 'conditional' if conditional else 'not_available' if negative else 'available' if positive else 'mentioned_unclear'
    return status, '\n'.join(evidence)[:2500]


ABSOLUTE_DATE = r'(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})'


def explicit_date(text, label):
    found = re.search(rf'(?i)(?:{label})\s*[:\-]?\s*({ABSOLUTE_DATE})', text)
    if not found:
        return pd.NaT, None
    raw = re.sub(r'(\d)(?:st|nd|rd|th)\b', r'\1', found[1], flags=re.I)
    return pd.to_datetime(raw, errors='coerce', utc=True), found[0]


def company_features(company, text):
    result = {'industry': None, 'industry_source': None, 'company_employees': None,
              'company_size': None, 'company_size_evidence': None, 'team_size': None}
    for label, pattern in [
        ('Finance', r'bank|financial|finance|credit union|capital markets'),
        ('Healthcare/Life sciences', r'health|pharma|biotech|hospital|medical'),
        ('Education', r'university|college|school'), ('Government', r'department of|city of|county of'),
        ('Energy/Utilities', r'energy|utilities|electric'),
    ]:
        if re.search(pattern, company, re.I):
            result.update(industry=label, industry_source='company_name_keyword'); break
    found = re.search(r'(?i)\b(?:over|more than|approximately|about)?\s*(\d{1,3}(?:,\d{3})+|\d{2,6})\+?\s+(?:employees|associates|team members)\b', text)
    if found:
        count = int(found[1].replace(',', ''))
        result.update(company_employees=count, company_size='small' if count < 50 else 'medium' if count < 250 else 'large' if count < 1000 else 'enterprise', company_size_evidence=found[0])
    team = re.search(r'(?i)\bteam of\s+(\d{1,3})\b', text)
    if team:
        result['team_size'] = int(team[1])
    return result


def extract_group(records, indices, rules, review):
    members = [records[i] for i in indices]
    # Prefer the most complete description; ties use stable input order.
    selected = max(members, key=lambda r: len(clean_text(r['job'].get('description'))))
    job = selected['job']
    description = clean_text(job.get('description'))
    extensions = list(dict.fromkeys(clean_text(x) for r in members for x in (r['job'].get('extensions') or [])))
    highlights = defaultdict(list)
    for r in members:
        for item in r['job'].get('job_highlights') or []:
            highlights[clean_text(item.get('title')).casefold()].extend(clean_text(x) for x in item.get('items', []) if x)
    highlights = {key: list(dict.fromkeys(value)) for key, value in highlights.items()}
    qualifications = highlights.get('qualifications', [])
    requirements = '\n'.join(qualifications) or description
    all_text = '\n'.join(extensions) + '\n' + description
    locations = sorted({clean_text(r['job'].get('location')) for r in members if r['job'].get('location')})
    titles = sorted({clean_text(r['job'].get('title')) for r in members if r['job'].get('title')})
    ids = sorted({clean_text(r['job'].get('job_id')) for r in members if r['job'].get('job_id')})
    platforms = sorted({re.sub(r'(?i)^via\s+', '', clean_text(r['job'].get('via'))) for r in members if r['job'].get('via')})
    selected_title = clean_text(job.get('title'))
    row = {
        'job_key': 'job-' + hashlib.sha256(('\n'.join(ids) or normalized(description)).encode()).hexdigest()[:20],
        'job_id': clean_text(job.get('job_id')) or None, 'job_ids': ids,
        'job_title': selected_title or None, 'alternate_titles': titles,
        'company_name': clean_text(job.get('company_name')) or None,
        'location_raw': clean_text(job.get('location')) or None, 'locations': locations,
        'posting_platform': re.sub(r'(?i)^via\s+', '', clean_text(job.get('via'))) or None,
        'posting_platforms': platforms, 'source_record_count': len(members),
        'search_queries': sorted({r['query'] for r in members if r['query']}),
        'source_files': sorted({r['source_file'] for r in members}),
        'description': description or None,
        'description_word_count': len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", description)),
        'description_char_count': len(description),
        'description_sha256': hashlib.sha256(normalized(description).encode()).hexdigest(),
        'qualifications': qualifications, 'responsibilities': highlights.get('responsibilities', []),
        'benefits_text': '\n'.join(highlights.get('benefits', [])) or None,
        'duplicate_match_rules': rules, 'possible_duplicate': review,
        'first_seen_utc': min(r['crawl_time'] for r in members),
        'last_seen_utc': max(r['crawl_time'] for r in members),
        'crawl_time_utc': selected['crawl_time'],
    }
    row.update(parse_location(row['location_raw']))
    row['multiple_locations'] = row['multiple_locations'] or len(locations) > 1
    for field in NUMERIC_COLUMNS + ['salary_period', 'salary_currency', 'salary_evidence', 'salary_source', 'salary_issue']:
        row.setdefault(field, None)
    row.update(salary_currency_assumed=None, salary_period_assumed=None, salary_multiple_ranges=None)
    candidates = [(str((job.get('detected_extensions') or {}).get('salary') or ''), 'detected_extensions')]
    candidates += [(label, 'extensions') for label in extensions]
    candidates += [('\n'.join(highlights.get('benefits', [])), 'benefits_highlight'), (description, 'description')]
    for text, source in candidates:
        parsed = parse_salary(text, source)
        if parsed:
            row.update(parsed); break
    detected = job.get('detected_extensions') or {}
    age = clean_text(detected.get('posted_at')) or next((x for x in extensions if re.fullmatch(r'(?i)(?:\d+\+?\s+(?:minute|hour|day|week|month|year)s? ago|today|yesterday|just posted|just now)', x)), None)
    row['posted_at_raw'] = age
    row['posted_date'], row['posted_date_precision'], row['posted_date_bound'] = posted_date(age, selected['crawl_time'])
    schedules = []
    for label, pattern in [('Full-time', r'\bfull[ -]time\b'), ('Part-time', r'\bpart[ -]time\b'),
                           ('Contract', r'\bcontract(?:or)?\b'), ('Internship', r'\bintern(?:ship)?\b'),
                           ('Temporary', r'\btemporary\b')]:
        if re.search(pattern, '\n'.join(extensions) + '\n' + str(detected.get('schedule_type') or ''), re.I):
            schedules.append(label)
    row['job_types'] = schedules
    row['job_type'] = schedules[0] if len(schedules) == 1 else 'Multiple' if schedules else None
    row.update(experience(requirements))
    row['experience_level'] = seniority(selected_title)
    row['education_degrees'] = matches(requirements, DEGREES)
    row['education_requirements_text'] = '\n'.join(s for s in clauses(requirements) if matches(s, DEGREES)) or None
    row['skills'] = matches(description, SKILLS)
    row['skill_count'] = len(row['skills'])
    row['certifications'] = matches(requirements, CERTIFICATIONS)
    row['soft_skills'] = matches(requirements, SOFT_SKILLS)
    benefits_context = '\n'.join(extensions + highlights.get('benefits', [])) or description
    row['benefits'] = sorted(name for name, pattern in BENEFITS.items() if any(
        re.search(pattern, s, re.I) and not re.search(r'(?i)\bno\b|\bnot eligible\b|\bnot offer\w*', s)
        for s in clauses(benefits_context)))
    row['company_values'] = matches(description, CULTURE)
    row['collaboration_tools'] = matches(description, TOOLS)
    row['work_arrangement'], row['work_arrangement_evidence'] = work_arrangement(selected_title, clean_text(job.get('location')), extensions, description)
    row['visa_sponsorship'], row['visa_sponsorship_evidence'] = sponsorship(description)
    row.update(company_features(row['company_name'] or '', description))
    travel = [int(m[1]) for s in clauses(description) if re.search(r'(?i)travel', s) for m in re.finditer(r'(\d{1,3})\s*%', s)]
    row['travel_percent_max'] = max(travel) if travel else None
    row['required_languages'] = sorted(name for name in ['English','Spanish','French','German','Mandarin','Japanese'] if any(
        re.search(rf'(?i)\b{name}\b', s) and re.search(r'(?i)fluen|proficien|required|bilingual|speak', s) for s in clauses(requirements)))
    evidence = {}
    for column, pattern in [('application_deadline', r'application deadline|apply by|applications? close|closing date'),
                            ('expected_start_date', r'expected start date|start date'),
                            ('posting_expiry_date', r'posting end date|posting expiry date|expiration date')]:
        row[column], evidence[column] = explicit_date(description, pattern)
    row['date_evidence'] = json.dumps({k:v for k,v in evidence.items() if v}, ensure_ascii=False)
    row['potential_non_job_posting'] = bool(re.search(r'(?i)paid surveys?|online surveys?|survey participants?', description) and re.search(r'(?i)earn haus|survey|participant', (row['company_name'] or '') + ' ' + selected_title))
    return row


def type_and_flag(frame):
    frame = frame.copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors='coerce').astype('Float64')
    for column in INTEGER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors='raise').astype('Int64')
    for column in DATE_COLUMNS + ['crawl_time_utc','first_seen_utc','last_seen_utc']:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors='coerce')
    boolean_columns = ['multiple_locations','possible_duplicate','salary_currency_assumed','salary_multiple_ranges',
                       'salary_period_assumed','potential_non_job_posting']
    for column in boolean_columns:
        frame[column] = frame[column].astype('boolean')
    typed = set(NUMERIC_COLUMNS + INTEGER_COLUMNS + DATE_COLUMNS + LIST_COLUMNS + boolean_columns +
                ['crawl_time_utc','first_seen_utc','last_seen_utc'])
    for column in frame.columns.difference(list(typed)):
        frame[column] = frame[column].astype('string')
    frame['salary_domain_outlier'] = (frame['salary_currency'].eq('USD') & ((frame['salary_annual_mid'] < 15000) | (frame['salary_annual_mid'] > 600000))).fillna(False)
    frame['experience_domain_outlier'] = ((frame['experience_years_min'] > 40) | (frame['experience_years_min'] < 0)).fillna(False)
    frame['description_length_outlier'] = (frame['description_word_count'] < 40) | (frame['description_word_count'] > 8000)
    for column, flag in [('salary_annual_mid','salary_iqr_outlier'),('experience_years_min','experience_iqr_outlier')]:
        sample = frame[column]
        if column.startswith('salary'):
            sample = sample.where(frame['salary_currency'] == 'USD')
        q1, q3 = sample.quantile([0.25,0.75])
        iqr = q3-q1
        frame[flag] = ((sample < q1-3*iqr) | (sample > q3+3*iqr)).fillna(False) if pd.notna(iqr) and iqr > 0 else False
    frame['quality_review'] = frame[['possible_duplicate','potential_non_job_posting','salary_multiple_ranges','salary_domain_outlier','salary_iqr_outlier','experience_domain_outlier','experience_iqr_outlier','description_length_outlier']].any(axis=1) | frame['salary_issue'].notna() | frame['work_arrangement'].eq('conflicting').fillna(False)
    return frame


def clean_dataset(data_dir):
    records = read_raw(data_dir)
    groups, edges, reviews = deduplicate(records)
    rows = [extract_group(records, indices, rules, review) for indices,rules,review in groups]
    frame = type_and_flag(pd.DataFrame(rows)).sort_values(['company_name','job_title','job_key'],kind='stable').reset_index(drop=True)
    membership = []
    for indices, rules, _ in groups:
        ids = sorted({clean_text(records[i]['job'].get('job_id')) for i in indices if records[i]['job'].get('job_id')})
        key = 'job-' + hashlib.sha256(('\n'.join(ids) or normalized(records[indices[0]]['job'].get('description'))).encode()).hexdigest()[:20]
        for i in indices:
            r = records[i]
            membership.append({'raw_index':i,'job_key':key,'job_id':r['job'].get('job_id'),'query':r['query'],
                               'source_file':r['source_file'],'source_index':r['source_index'],'match_rules':'; '.join(rules)})
    return frame, records, pd.DataFrame(membership), pd.DataFrame(edges), pd.DataFrame(reviews)


def export_csv(frame, output):
    serial = frame.copy()
    for column in LIST_COLUMNS:
        serial[column] = serial[column].map(lambda value: json.dumps(value,ensure_ascii=False))
    for column in DATE_COLUMNS:
        serial[column] = serial[column].dt.strftime('%Y-%m-%d')
    for column in ['crawl_time_utc','first_seen_utc','last_seen_utc']:
        serial[column] = serial[column].map(lambda value: value.isoformat() if pd.notna(value) else None)
    serial.to_csv(output,index=False,encoding='utf-8',na_rep='',float_format='%.4f')
    return serial


def quality_tables(frame):
    rows=[]
    for column in frame:
        missing = frame[column].map(len).eq(0) if column in LIST_COLUMNS else frame[column].isna()
        rows.append({'column':column,'dtype':str(frame[column].dtype),'observed_rows':int((~missing).sum()),
                     'missing_or_no_mention_rows':int(missing.sum()),'coverage_percent':round(100*(~missing).mean(),2)})
    return pd.DataFrame(rows).sort_values(['coverage_percent','column']), frame.loc[frame['quality_review']].copy()
