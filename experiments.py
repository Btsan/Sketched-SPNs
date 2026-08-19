ALL_EXPERIMENTS = ['stats-ceb', 'job-light', 'job-light-ranges', 'job', 'job-complex', 'stats-sqlstorm']

def get_date_cols(experiment:str):
    experiment = experiment.lower()
    dates = dict()
    if experiment == 'stats-ceb':
        dates = {'badges': {'Date',},
                 'comments': {'CreationDate',},
                 'postHistory': {'CreationDate',},
                 'postLinks': {'CreationDate',},
                 'posts': {'CreationDate',},
                 'users': {'CreationDate',},
                 'votes': {'CreationDate',},}
    if experiment == 'stats-sqlstorm':
        dates = {'badges': {'Date',},
                 'comments': {'CreationDate',},
                 'postHistory': {'CreationDate',},
                 'postLinks': {'CreationDate',},
                 'posts': {'CreationDate',},
                 'users': {'CreationDate',},
                 'votes': {'CreationDate',},}
    return dates

def get_string_cols(experiment:str):
    experiment = experiment.lower()
    strings = dict()
    if experiment == 'job':
        strings = {'title': {'title',},
                 'movie_companies': {'note',},
                 'keyword': {'keyword',},
                 'company_name': {'name',},
                 'cast_info': {'note',},
                 'movie_info': {'info', 'note'},
                 'name': {'gender', 'name', 'name_pcode_cf'},
                 'aka_name': {'name'},
                 'char_name': {'name'},
                 'comp_cast_type': {'kind',},
                 'link_type': {'link',},}
    return strings


def get_range_intervals(experiment:str):
    experiment = experiment.lower()
    intervals = dict()

    # note: dates are treated as nanoseconds (1e-9 seconds)
    TIMESTAMP_INTERVAL_PRESET = (10**9 * 3600, # hours
                                 10**9 * 3600 * 2, # 2 hours
                                 10**9 * 3600 * 4, # 4 hours
                                 10**9 * 3600 * 12, # 12 hours
                                 10**9 * 3600 * 24, # days
                                 10**9 * 3600 * 24 * 7, # weeks
                                 10**9 * 3600 * 24 * 14, # 2 weeks
                                 10**9 * 3600 * 24 * 28, # months
                                 10**9 * 3600 * 24 * 28 * 13, # years
                                 )
    
    # note: if col_types order changes, old rdc features are invalidated
    if experiment == 'stats-ceb':
        # ideally there are intevals for all continuous attributes
        # intervals should be multiples of the next smallest interval
        # this effectively bins the data into intervals of the given granularity
        # has a huge impact on the performance of the SPN
        # the more intervals (with finer granularity) the more accurate (and larger) the model
        intervals = {'badges': {'Date': TIMESTAMP_INTERVAL_PRESET,},
                    'comments': {'CreationDate': TIMESTAMP_INTERVAL_PRESET, 
                                  'Score': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},
                    'postHistory': {'CreationDate': TIMESTAMP_INTERVAL_PRESET}, 
                    'postLinks': {'CreationDate': TIMESTAMP_INTERVAL_PRESET},
                    'posts': {'CreationDate': TIMESTAMP_INTERVAL_PRESET,
                              'Score': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'ViewCount': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'AnswerCount': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'CommentCount': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'FavoriteCount': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},
                    'tags': {'Count': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},
                    'users': {'CreationDate': TIMESTAMP_INTERVAL_PRESET,
                              'Reputation': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'Views': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'UpVotes': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'DownVotes': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},
                    'votes': {'CreationDate': TIMESTAMP_INTERVAL_PRESET,
                              'BountyAmount': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},}
    elif experiment == 'stats-sqlstorm':
        # ideally there are intevals for all continuous attributes
        # intervals should be multiples of the next smallest interval
        # this effectively bins the data into intervals of the given granularity
        # has a huge impact on the performance of the SPN
        # the more intervals (with finer granularity) the more accurate (and larger) the model
        intervals = {'badges': {'Date': TIMESTAMP_INTERVAL_PRESET,},
                    'comments': {'CreationDate': TIMESTAMP_INTERVAL_PRESET, 
                                  'Score': (1, 2, 4,),},
                    'postHistory': {'CreationDate': TIMESTAMP_INTERVAL_PRESET,
                                    'PostHistoryTypeId': (1, 2, 4)}, 
                    'postLinks': {'LinkTypeId': (1, 2, 4)},
                    'posts': {'CreationDate': TIMESTAMP_INTERVAL_PRESET,
                              'Score': (1, 2, 4, 8,),
                              'ViewCount': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),},
                    'tags': {'Count': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},
                    'users': {'CreationDate': TIMESTAMP_INTERVAL_PRESET,
                              'Reputation': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'Views': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'UpVotes': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'DownVotes': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},
                    'votes': {'CreationDate': TIMESTAMP_INTERVAL_PRESET,
                              'BountyAmount': (1, 2, 4, 8, 16, 32, 64, 128),},}
    elif experiment == 'job-light':
        intervals = {'title': {'production_year': (1, 2, 4, 8 ,16, 32),
                               'episode_nr': (1, 2, 4, 8, 16, 32)},
                     'name': {'name_pcode_cf': (1, 2, 4)}}
    elif experiment == 'job-light-ranges':
        intervals = {'title': {'production_year': (1, 2, 4, 8 ,16, 32),
                               'episode_nr': (1, 2, 4, 8, 16, 32)}}
    elif experiment == 'job':
        intervals = {'title': {'production_year': (1, 2, 4, 8 ,16, 32),
                               'episode_nr': (1, 2, 4, 8, 16, 32)}}
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    return intervals

def get_tables_cols(experiment:str):
    experiment = experiment.lower()
    tables = dict()

    # note: discrete is intended as non-ordinal
    # the distinction is used for correlation metrics (e.g., RDC)
    # categorizing text attributes as discrete implies that only equality comparisons are meaningful
    # I.e., mislabeling ordinal attributes as discrete causes loss of information, but the reverse may lead to incorrect correlations
    DISCRETE = 'DISCRETE'
    CONTINUOUS = 'CONTINUOUS'
    # For best results, only label attributes as CONTINUOUS if your workload uses them for inequalities or LIKE predicates

    # note: if col_types order changes, old rdc features are invalidated
    if experiment == 'stats-ceb':
        tables = {'badges': {'names': None,
                             'col_types': {'Id': DISCRETE,
                                         'UserId': DISCRETE,
                                         'Date': CONTINUOUS,},
                             'keys': {'UserId'}},
                  'comments': {'names': None,
                               'col_types': {'Id': DISCRETE,
                                           'PostId': DISCRETE,
                                           'Score': CONTINUOUS,
                                           'CreationDate': CONTINUOUS,
                                           'UserId': DISCRETE,},
                               'keys': {'UserId', 'PostId'}},
                  'postHistory': {'names': None,
                                  'col_types': {'Id': DISCRETE,
                                              'PostHistoryTypeId': DISCRETE,
                                              'PostId': DISCRETE,
                                              'CreationDate': CONTINUOUS,
                                              'UserId': DISCRETE,},
                                  'keys': {'UserId', 'PostId'}},
                  'postLinks': {'names': None,
                                'col_types': {'Id': DISCRETE,
                                            'CreationDate': CONTINUOUS,
                                            'PostId': DISCRETE,
                                            'RelatedPostId': DISCRETE,
                                            'LinkTypeId': DISCRETE,},
                                'keys': {'RelatedPostId', 'PostId'}},
                  'posts': {'names': None,
                            'col_types': {'Id': DISCRETE,
                                        'PostTypeId': DISCRETE,
                                        'CreationDate': CONTINUOUS,
                                        'Score': CONTINUOUS,
                                        'ViewCount': CONTINUOUS,
                                        'OwnerUserId': DISCRETE,
                                        'AnswerCount': CONTINUOUS,
                                        'CommentCount': CONTINUOUS,
                                        'FavoriteCount': CONTINUOUS,
                                        'LastEditorUserId': DISCRETE,}, 
                            'keys': {'OwnerUserId', 'Id'}},
                  'tags': {'names': None,
                           'col_types': {'Id': DISCRETE,
                                       'Count': CONTINUOUS,
                                       'ExcerptPostId': CONTINUOUS,},
                           'keys': {'ExcerptPostId'}},
                  'users': {'names': None,
                            'col_types': {'Id': DISCRETE,
                                        'Reputation': CONTINUOUS,
                                        'CreationDate': CONTINUOUS,
                                        'Views': CONTINUOUS,
                                        'UpVotes': CONTINUOUS,
                                        'DownVotes': CONTINUOUS,},
                            'keys': {'Id'}},
                  'votes': {'names': None,
                            'col_types': {'Id': DISCRETE,
                                        'PostId': DISCRETE,
                                        'VoteTypeId': DISCRETE,
                                        'CreationDate': CONTINUOUS,
                                        'UserId': DISCRETE,
                                        'BountyAmount': CONTINUOUS},
                            'keys': {'UserId', 'PostId'}}}
    elif experiment == 'job-light':
        tables = {'title': {'names': ['id', 'title', 'imdb_index', 'kind_id', 'production_year',
                                'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr',
                                'episode_nr', 'series_years', 'md5sum'],
                            'col_types': {'id': DISCRETE,
                                          'kind_id': DISCRETE,
                                          'production_year': CONTINUOUS},
                            'keys': {'id'},},
                'movie_info': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                'col_types': {'info_type_id': DISCRETE,
                                              'movie_id': DISCRETE},
                                'keys': {'movie_id'},},
                'movie_companies': {'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                                    'col_types': {'company_type_id': DISCRETE,
                                                  'company_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id'},},
                'movie_info_idx': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'info_type_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id'},},
                'movie_keyword': {'names': ['id', 'movie_id', 'keyword_id'],
                                    'col_types': {'keyword_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id'},},
                'cast_info': {'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                            'col_types': {'role_id': DISCRETE,
                                          'movie_id': DISCRETE},
                            'keys': {'movie_id'}}}
    elif experiment == 'job-light-ranges':
        tables = {'title': {'names': ['id', 'title', 'imdb_index', 'kind_id', 'production_year',
                                'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr',
                                'episode_nr', 'series_years', 'md5sum'],
                            'col_types': {'id': DISCRETE,
                                          'kind_id': DISCRETE,
                                          'production_year': CONTINUOUS,
                                          'title': CONTINUOUS, # LIKE
                                          'episode_nr': CONTINUOUS},
                            'keys': {'id', 'kind_id'},},
                'movie_companies': {'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                                    'col_types': {'company_type_id': DISCRETE,
                                                  'company_id': DISCRETE,
                                                  'movie_id': DISCRETE,
                                                  'note': CONTINUOUS}, # LIKE
                                    'keys': {'movie_id', 'company_id', 'company_type_id'},},
                'keyword': {'names': ['id', 'keyword', 'phonetic_code'],
                                    'col_types': {'id': DISCRETE,
                                                  'keyword': CONTINUOUS,}, # LIKE
                                    'keys': {'id'},},
                'movie_keyword': {'names': ['id', 'movie_id', 'keyword_id'],
                                    'col_types': {'keyword_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'keyword_id'},},
                'company_name': {'names': ['id', 'name', 'country_code', 'imdb_id',
                                           'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'country_code': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'info_type': {'names': ['id', 'info'],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE},
                                    'keys': {'id'},},
                'cast_info': {'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                            'col_types': {'role_id': DISCRETE,
                                          'movie_id': DISCRETE,
                                          'person_id': DISCRETE,
                                          'person_role_id': DISCRETE,
                                          'note': CONTINUOUS}, # LIKE
                            'keys': {'movie_id', 'person_id', 'person_role_id', 'role_id'}},
                'movie_info': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                'col_types': {'info_type_id': DISCRETE,
                                              'movie_id': DISCRETE,
                                              'info': CONTINUOUS, # LIKE
                                              'note': CONTINUOUS}, # LIKE
                                'keys': {'movie_id', 'info_type_id'},},
                'name': {'names': ['id', 'name', 'imdb_index', 'imdb_id', 'gender',
                                   'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'gender': CONTINUOUS, # LIKE
                                                  'name': CONTINUOUS, # LIKE
                                                  'name_pcode_cf': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'movie_info_idx': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'info_type_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'info_type_id'},},
                'company_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id'},},
                'kind_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id'},},
                'aka_name': {'names': ['id', 'person_id', 'name', 'imdb_index',
                                       'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'person_id': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'person_id'},},
                'char_name': {'names': ['id', 'name', 'imdb_index', 'imdb_id',
                                        'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'complete_cast': {'names': ['id', 'movie_id', 'subject_id', 'status_id'],
                                    'col_types': {'movie_id': DISCRETE,
                                                  'status_id': DISCRETE,
                                                  'subject_id': DISCRETE},
                                    'keys': {'movie_id', 'status_id', 'subject_id'},},
                'comp_cast_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'role_type': {'names': ['id', 'role'],
                                    'col_types': {'id': DISCRETE,
                                                  'role': DISCRETE},
                                    'keys': {'id'},},
                'link_type': {'names': ['id', 'link'],
                                    'col_types': {'id': DISCRETE,
                                                  'link': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'movie_link': {'names': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
                                    'col_types': {'link_type_id': DISCRETE,
                                                  'movie_id': DISCRETE,
                                                  'linked_movie_id': DISCRETE},
                                    'keys': {'link_type_id', 'movie_id', 'linked_movie_id'},},
                'person_info': {'names': ['id', 'person_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'person_id': DISCRETE,
                                                  'info_type_id': DISCRETE,
                                                  'note': DISCRETE},
                                    'keys': {'person_id', 'info_type_id'},},
                'aka_title': {'names': ['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                                        'production_year', 'phonetic_code', 'episode_of_id',
                                        'season_nr', 'episode_nr', 'note', 'md5sum'],
                                    'col_types': {'movie_id': DISCRETE},
                                    'keys': {'movie_id'},},
                }
    elif experiment == 'job':
        tables = {'title': {'names': ['id', 'title', 'imdb_index', 'kind_id', 'production_year',
                                'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr',
                                'episode_nr', 'series_years', 'md5sum'],
                            'col_types': {'id': DISCRETE,
                                          'kind_id': DISCRETE,
                                          'production_year': CONTINUOUS,
                                          'title': CONTINUOUS, # LIKE
                                          'episode_nr': CONTINUOUS},
                            'keys': {'id', 'kind_id'},},
                'movie_companies': {'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                                    'col_types': {'company_type_id': DISCRETE,
                                                  'company_id': DISCRETE,
                                                  'movie_id': DISCRETE,
                                                  'note': CONTINUOUS}, # LIKE
                                    'keys': {'movie_id', 'company_id', 'company_type_id'},},
                'keyword': {'names': ['id', 'keyword', 'phonetic_code'],
                                    'col_types': {'id': DISCRETE,
                                                  'keyword': CONTINUOUS,}, # LIKE
                                    'keys': {'id'},},
                'movie_keyword': {'names': ['id', 'movie_id', 'keyword_id'],
                                    'col_types': {'keyword_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'keyword_id'},},
                'company_name': {'names': ['id', 'name', 'country_code', 'imdb_id',
                                           'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'country_code': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'info_type': {'names': ['id', 'info'],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE},
                                    'keys': {'id', 'info'},},
                'cast_info': {'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                            'col_types': {'role_id': DISCRETE,
                                          'movie_id': DISCRETE,
                                          'person_id': DISCRETE,
                                          'person_role_id': DISCRETE,
                                          'note': CONTINUOUS}, # LIKE
                            'keys': {'movie_id', 'person_id', 'person_role_id', 'role_id'}},
                'movie_info': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                'col_types': {'info_type_id': DISCRETE,
                                              'movie_id': DISCRETE,
                                              'info': CONTINUOUS, # LIKE
                                              'note': CONTINUOUS}, # LIKE
                                'keys': {'movie_id', 'info_type_id'},},
                'name': {'names': ['id', 'name', 'imdb_index', 'imdb_id', 'gender',
                                   'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'gender': CONTINUOUS, # LIKE
                                                  'name': CONTINUOUS, # LIKE
                                                  'name_pcode_cf': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'movie_info_idx': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'info_type_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'info_type_id'},},
                'company_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id'},},
                'kind_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id', 'kind'},},
                'aka_name': {'names': ['id', 'person_id', 'name', 'imdb_index',
                                       'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'person_id': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'person_id'},},
                'char_name': {'names': ['id', 'name', 'imdb_index', 'imdb_id',
                                        'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'complete_cast': {'names': ['id', 'movie_id', 'subject_id', 'status_id'],
                                    'col_types': {'movie_id': DISCRETE,
                                                  'status_id': DISCRETE,
                                                  'subject_id': DISCRETE},
                                    'keys': {'movie_id', 'status_id', 'subject_id'},},
                'comp_cast_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'role_type': {'names': ['id', 'role'],
                                    'col_types': {'id': DISCRETE,
                                                  'role': DISCRETE},
                                    'keys': {'id'},},
                'link_type': {'names': ['id', 'link'],
                                    'col_types': {'id': DISCRETE,
                                                  'link': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'movie_link': {'names': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
                                    'col_types': {'link_type_id': DISCRETE,
                                                  'movie_id': DISCRETE,
                                                  'linked_movie_id': DISCRETE},
                                    'keys': {'link_type_id', 'movie_id', 'linked_movie_id'},},
                'person_info': {'names': ['id', 'person_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'person_id': DISCRETE,
                                                  'info_type_id': DISCRETE,
                                                  'note': DISCRETE},
                                    'keys': {'person_id', 'info_type_id'},},
                'aka_title': {'names': ['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                                        'production_year', 'phonetic_code', 'episode_of_id',
                                        'season_nr', 'episode_nr', 'note', 'md5sum'],
                                    'col_types': {'movie_id': DISCRETE},
                                    'keys': {'movie_id'},},
                }
    elif experiment == 'job-complex':
        tables = {'title': {'names': ['id', 'title', 'imdb_index', 'kind_id', 'production_year',
                                'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr',
                                'episode_nr', 'series_years', 'md5sum'],
                            'col_types': {'id': DISCRETE,
                                          'kind_id': DISCRETE,
                                          'production_year': CONTINUOUS,
                                          'title': CONTINUOUS, # LIKE
                                          'episode_nr': DISCRETE},
                            'keys': {'id', 'kind_id', 'imdb_index', 'episode_nr'},},
                'movie_companies': {'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                                    'col_types': {'company_type_id': DISCRETE,
                                                  'company_id': DISCRETE,
                                                  'movie_id': DISCRETE,
                                                  'note': CONTINUOUS}, # LIKE
                                    'keys': {'movie_id', 'company_id', 'company_type_id'},},
                'keyword': {'names': ['id', 'keyword', 'phonetic_code'],
                                    'col_types': {'id': DISCRETE,
                                                  'keyword': DISCRETE,
                                                  'phonetic_code': DISCRETE},
                                    'keys': {'id', 'phonetic_code'},},
                'movie_keyword': {'names': ['id', 'movie_id', 'keyword_id'],
                                    'col_types': {'keyword_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'keyword_id'},},
                'company_name': {'names': ['id', 'name', 'country_code', 'imdb_id',
                                           'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'name_pcode_sf': DISCRETE,
                                                  'country_code': DISCRETE,
                                                  'name_pcode_nf': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'id', 'name_pcode_sf', 'name_pcode_nf'},},
                'info_type': {'names': ['id', 'info'],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE},
                                    'keys': {'id'},},
                'cast_info': {'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                            'col_types': {'nr_order': DISCRETE,
                                          'movie_id': DISCRETE,
                                          'person_id': DISCRETE,
                                          'person_role_id': DISCRETE,
                                          'note': DISCRETE},
                            'keys': {'movie_id', 'person_id', 'person_role_id', 'nr_order'}},
                'movie_info': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                'col_types': {'info_type_id': DISCRETE,
                                              'movie_id': DISCRETE,
                                              'info': CONTINUOUS, # LIKE
                                              'note': CONTINUOUS}, # LIKE
                                'keys': {'movie_id', 'info_type_id'},},
                'name': {'names': ['id', 'name', 'imdb_index', 'imdb_id', 'gender',
                                   'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'gender': DISCRETE,
                                                  'name': CONTINUOUS,
                                                  'name_pcode_cf': DISCRETE,
                                                  'name_pcode_nf': DISCRETE,
                                                  'surname_pcode': DISCRETE,
                                                  'imdb_index': DISCRETE},
                                    'keys': {'id', 'name', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'imdb_index'},},
                'movie_info_idx': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'info_type_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'info_type_id'},},
                'company_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id'},},
                'kind_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id'},},
                'aka_name': {'names': ['id', 'person_id', 'name', 'imdb_index',
                                       'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'name_pcode_nf': DISCRETE,
                                                  'name_pcode_cf': DISCRETE,
                                                  'imdb_index': DISCRETE,
                                                  'surname_pcode': DISCRETE,},
                                    'keys': {'name_pcode_nf', 'name_pcode_cf', 'imdb_index', 'surname_pcode'},},
                'char_name': {'names': ['id', 'name', 'imdb_index', 'imdb_id',
                                        'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'name': CONTINUOUS, # LIKE but also join key
                                                  'surname_pcode': DISCRETE,
                                                  'name_pcode_nf': DISCRETE,}, 
                                    'keys': {'id', 'name_pcode_nf', 'surname_pcode', 'name'},},
                'complete_cast': {'names': ['id', 'movie_id', 'subject_id', 'status_id'],
                                    'col_types': {'movie_id': DISCRETE,
                                                  'status_id': DISCRETE,
                                                  'subject_id': DISCRETE},
                                    'keys': {'movie_id', 'status_id', 'subject_id'},},
                'comp_cast_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'link_type': {'names': ['id', 'link'],
                                    'col_types': {'id': DISCRETE,
                                                  'link': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'movie_link': {'names': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
                                    'col_types': {'link_type_id': DISCRETE,
                                                  'movie_id': DISCRETE,},
                                    'keys': {'link_type_id', 'movie_id'},},
                'aka_title': {'names': ['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                                        'production_year', 'phonetic_code', 'episode_of_id',
                                        'season_nr', 'episode_nr', 'note', 'md5sum'],
                                    'col_types': {'movie_id': DISCRETE,
                                                  'imdb_index': DISCRETE,},
                                    'keys': {'movie_id', 'imdb_index'},},
                }
    elif experiment == 'stats-sqlstorm':
        tables = {'badges': {'names': None,
                             'col_types': {'UserId': DISCRETE,
                                         'Date': CONTINUOUS,},
                             'keys': {'UserId'}},
                  'comments': {'names': None,
                               'col_types': {'PostId': DISCRETE,
                                           'Score': CONTINUOUS,
                                           'CreationDate': CONTINUOUS,},
                               'keys': {'PostId'}},
                  'postHistory': {'names': None,
                                  'col_types': {'Id': DISCRETE,
                                              'PostHistoryTypeId': DISCRETE,
                                              'PostId': DISCRETE,
                                              'CreationDate': CONTINUOUS,
                                              'UserId': DISCRETE,},
                                  'keys': {'UserId', 'PostId'}},
                  'postLinks': {'names': None,
                                'col_types': {'PostId': DISCRETE,
                                            'LinkTypeId': DISCRETE,},
                                'keys': {'PostId'}},
                  'posts': {'names': None,
                            'col_types': {'Id': DISCRETE,
                                        'PostTypeId': DISCRETE,
                                        'CreationDate': CONTINUOUS,
                                        'Score': CONTINUOUS,
                                        'ViewCount': CONTINUOUS,
                                        'OwnerUserId': DISCRETE,}, 
                            'keys': {'OwnerUserId', 'Id'}},
                  'tags': {'names': None,
                           'col_types': {'Id': DISCRETE,
                                       'Count': CONTINUOUS,
                                       'ExcerptPostId': CONTINUOUS,},
                           'keys': {'ExcerptPostId'}},
                  'users': {'names': None,
                            'col_types': {'Id': DISCRETE,
                                        'Reputation': CONTINUOUS,
                                        'Views': CONTINUOUS,},
                            'keys': {'Id'}},
                  'votes': {'names': None,
                            'col_types': {'PostId': DISCRETE,
                                        'VoteTypeId': DISCRETE,
                                        'CreationDate': CONTINUOUS,
                                        'BountyAmount': CONTINUOUS},
                            'keys': {'PostId'}}}
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    # validate other join keys
    for _, meta in tables.items():
        assert meta['keys'].intersection(meta['col_types']), f"{meta['keys']} must intersect {meta['col_types']}"
        
    return tables

def get_config(experiment:str):
    experiment = experiment.lower()
    dates = dict()
    intervals = dict()
    tables = dict()

    # note: discrete is intended as non-ordinal
    # the distinction is used for correlation metrics (e.g., RDC)
    # categorizing text attributes as discrete implies that only equality comparisons are meaningful
    # I.e., mislabeling ordinal attributes as discrete causes loss of information, but the reverse may lead to incorrect correlations
    DISCRETE = 'DISCRETE'
    CONTINUOUS = 'CONTINUOUS'
    # For best results, only label attributes as CONTINUOUS if your workload uses them for inequalities or LIKE predicates

    # note: dates are treated as nanoseconds (1e-9 seconds)
    TIMESTAMP_INTERVAL_PRESET = (10**9 * 3600, # hours
                                 10**9 * 3600 * 2, # 2 hours
                                 10**9 * 3600 * 4, # 4 hours
                                 10**9 * 3600 * 12, # 12 hours
                                 10**9 * 3600 * 24, # days
                                 10**9 * 3600 * 24 * 7, # weeks
                                 10**9 * 3600 * 24 * 14, # 2 weeks
                                 10**9 * 3600 * 24 * 28, # months
                                 10**9 * 3600 * 24 * 28 * 13, # years
                                 )

    # note: if col_types order changes, old rdc features are invalidated
    if experiment == 'stats-ceb':
        dates = {'badges': {'Date',},
                 'comments': {'CreationDate',},
                 'postHistory': {'CreationDate',},
                 'postLinks': {'CreationDate',},
                 'posts': {'CreationDate',},
                 'users': {'CreationDate',},
                 'votes': {'CreationDate',},}
        # ideally there are intevals for all continuous attributes
        # intervals should be multiples of the next smallest interval
        # this effectively bins the data into intervals of the given granularity
        # has a huge impact on the performance of the SPN
        # the more intervals (with finer granularity) the more accurate (and larger) the model
        intervals = {'badges': {'Date': TIMESTAMP_INTERVAL_PRESET,},
                    'comments': {'CreationDate': TIMESTAMP_INTERVAL_PRESET, 
                                  'Score': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},
                    'postHistory': {'CreationDate': TIMESTAMP_INTERVAL_PRESET}, 
                    'postLinks': {'CreationDate': TIMESTAMP_INTERVAL_PRESET},
                    'posts': {'CreationDate': TIMESTAMP_INTERVAL_PRESET,
                              'Score': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'ViewCount': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'AnswerCount': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'CommentCount': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'FavoriteCount': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},
                    'tags': {'Count': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},
                    'users': {'CreationDate': TIMESTAMP_INTERVAL_PRESET,
                              'Reputation': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'Views': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'UpVotes': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                              'DownVotes': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},
                    'votes': {'CreationDate': TIMESTAMP_INTERVAL_PRESET,
                              'BountyAmount': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),},}
        tables = {'badges': {'names': None,
                             'col_types': {'Id': DISCRETE,
                                         'UserId': DISCRETE,
                                         'Date': CONTINUOUS,},
                             'keys': {'UserId'}},
                  'comments': {'names': None,
                               'col_types': {'Id': DISCRETE,
                                           'PostId': DISCRETE,
                                           'Score': CONTINUOUS,
                                           'CreationDate': CONTINUOUS,
                                           'UserId': DISCRETE,},
                               'keys': {'UserId', 'PostId'}},
                  'postHistory': {'names': None,
                                  'col_types': {'Id': DISCRETE,
                                              'PostHistoryTypeId': DISCRETE,
                                              'PostId': DISCRETE,
                                              'CreationDate': CONTINUOUS,
                                              'UserId': DISCRETE,},
                                  'keys': {'UserId', 'PostId'}},
                  'postLinks': {'names': None,
                                'col_types': {'Id': DISCRETE,
                                            'CreationDate': CONTINUOUS,
                                            'PostId': DISCRETE,
                                            'RelatedPostId': DISCRETE,
                                            'LinkTypeId': DISCRETE,},
                                'keys': {'RelatedPostId', 'PostId'}},
                  'posts': {'names': None,
                            'col_types': {'Id': DISCRETE,
                                        'PostTypeId': DISCRETE,
                                        'CreationDate': CONTINUOUS,
                                        'Score': CONTINUOUS,
                                        'ViewCount': CONTINUOUS,
                                        'OwnerUserId': DISCRETE,
                                        'AnswerCount': CONTINUOUS,
                                        'CommentCount': CONTINUOUS,
                                        'FavoriteCount': CONTINUOUS,
                                        'LastEditorUserId': DISCRETE,}, 
                            'keys': {'OwnerUserId', 'Id'}},
                  'tags': {'names': None,
                           'col_types': {'Id': DISCRETE,
                                       'Count': CONTINUOUS,
                                       'ExcerptPostId': CONTINUOUS,},
                           'keys': {'ExcerptPostId'}},
                  'users': {'names': None,
                            'col_types': {'Id': DISCRETE,
                                        'Reputation': CONTINUOUS,
                                        'CreationDate': CONTINUOUS,
                                        'Views': CONTINUOUS,
                                        'UpVotes': CONTINUOUS,
                                        'DownVotes': CONTINUOUS,},
                            'keys': {'Id'}},
                  'votes': {'names': None,
                            'col_types': {'Id': DISCRETE,
                                        'PostId': DISCRETE,
                                        'VoteTypeId': DISCRETE,
                                        'CreationDate': CONTINUOUS,
                                        'UserId': DISCRETE,
                                        'BountyAmount': CONTINUOUS},
                            'keys': {'UserId', 'PostId'}}}
    elif experiment == 'job-light':
        dates = dict()
        intervals = {'title': {'production_year': (1, 2, 4, 8 ,16, 32),
                               'episode_nr': (1, 2, 4, 8, 16, 32)},
                     'name': {'name_pcode_cf': (1, 2, 4)}}
        tables = {'title': {'names': ['id', 'title', 'imdb_index', 'kind_id', 'production_year',
                                'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr',
                                'episode_nr', 'series_years', 'md5sum'],
                            'col_types': {'id': DISCRETE,
                                          'kind_id': DISCRETE,
                                          'production_year': CONTINUOUS},
                            'keys': {'id'},},
                'movie_info': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                'col_types': {'info_type_id': DISCRETE,
                                              'movie_id': DISCRETE},
                                'keys': {'movie_id'},},
                'movie_companies': {'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                                    'col_types': {'company_type_id': DISCRETE,
                                                  'company_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id'},},
                'movie_info_idx': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'info_type_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id'},},
                'movie_keyword': {'names': ['id', 'movie_id', 'keyword_id'],
                                    'col_types': {'keyword_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id'},},
                'cast_info': {'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                            'col_types': {'role_id': DISCRETE,
                                          'movie_id': DISCRETE},
                            'keys': {'movie_id'}}}
    elif experiment == 'job-light-ranges':
        dates = dict()
        intervals = {'title': {'production_year': (1, 2, 4, 8 ,16, 32),
                               'episode_nr': (1, 2, 4, 8, 16, 32)}}
        tables = {'title': {'names': ['id', 'title', 'imdb_index', 'kind_id', 'production_year',
                                'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr',
                                'episode_nr', 'series_years', 'md5sum'],
                            'col_types': {'id': DISCRETE,
                                          'kind_id': DISCRETE,
                                          'production_year': CONTINUOUS,
                                          'title': CONTINUOUS, # LIKE
                                          'episode_nr': CONTINUOUS},
                            'keys': {'id', 'kind_id'},},
                'movie_companies': {'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                                    'col_types': {'company_type_id': DISCRETE,
                                                  'company_id': DISCRETE,
                                                  'movie_id': DISCRETE,
                                                  'note': CONTINUOUS}, # LIKE
                                    'keys': {'movie_id', 'company_id', 'company_type_id'},},
                'keyword': {'names': ['id', 'keyword', 'phonetic_code'],
                                    'col_types': {'id': DISCRETE,
                                                  'keyword': CONTINUOUS,}, # LIKE
                                    'keys': {'id'},},
                'movie_keyword': {'names': ['id', 'movie_id', 'keyword_id'],
                                    'col_types': {'keyword_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'keyword_id'},},
                'company_name': {'names': ['id', 'name', 'country_code', 'imdb_id',
                                           'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'country_code': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'info_type': {'names': ['id', 'info'],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE},
                                    'keys': {'id'},},
                'cast_info': {'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                            'col_types': {'role_id': DISCRETE,
                                          'movie_id': DISCRETE,
                                          'person_id': DISCRETE,
                                          'person_role_id': DISCRETE,
                                          'note': CONTINUOUS}, # LIKE
                            'keys': {'movie_id', 'person_id', 'person_role_id', 'role_id'}},
                'movie_info': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                'col_types': {'info_type_id': DISCRETE,
                                              'movie_id': DISCRETE,
                                              'info': CONTINUOUS, # LIKE
                                              'note': CONTINUOUS}, # LIKE
                                'keys': {'movie_id', 'info_type_id'},},
                'name': {'names': ['id', 'name', 'imdb_index', 'imdb_id', 'gender',
                                   'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'gender': CONTINUOUS, # LIKE
                                                  'name': CONTINUOUS, # LIKE
                                                  'name_pcode_cf': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'movie_info_idx': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'info_type_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'info_type_id'},},
                'company_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id'},},
                'kind_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id'},},
                'aka_name': {'names': ['id', 'person_id', 'name', 'imdb_index',
                                       'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'person_id': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'person_id'},},
                'char_name': {'names': ['id', 'name', 'imdb_index', 'imdb_id',
                                        'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'complete_cast': {'names': ['id', 'movie_id', 'subject_id', 'status_id'],
                                    'col_types': {'movie_id': DISCRETE,
                                                  'status_id': DISCRETE,
                                                  'subject_id': DISCRETE},
                                    'keys': {'movie_id', 'status_id', 'subject_id'},},
                'comp_cast_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'role_type': {'names': ['id', 'role'],
                                    'col_types': {'id': DISCRETE,
                                                  'role': DISCRETE},
                                    'keys': {'id'},},
                'link_type': {'names': ['id', 'link'],
                                    'col_types': {'id': DISCRETE,
                                                  'link': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'movie_link': {'names': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
                                    'col_types': {'link_type_id': DISCRETE,
                                                  'movie_id': DISCRETE,
                                                  'linked_movie_id': DISCRETE},
                                    'keys': {'link_type_id', 'movie_id', 'linked_movie_id'},},
                'person_info': {'names': ['id', 'person_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'person_id': DISCRETE,
                                                  'info_type_id': DISCRETE,
                                                  'note': DISCRETE},
                                    'keys': {'person_id', 'info_type_id'},},
                'aka_title': {'names': ['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                                        'production_year', 'phonetic_code', 'episode_of_id',
                                        'season_nr', 'episode_nr', 'note', 'md5sum'],
                                    'col_types': {'movie_id': DISCRETE},
                                    'keys': {'movie_id'},},
                }
    elif experiment == 'job':
        dates = dict()
        intervals = {'title': {'production_year': (1, 2, 4, 8 ,16, 32),
                               'episode_nr': (1, 2, 4, 8, 16, 32)}}
        tables = {'title': {'names': ['id', 'title', 'imdb_index', 'kind_id', 'production_year',
                                'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr',
                                'episode_nr', 'series_years', 'md5sum'],
                            'col_types': {'id': DISCRETE,
                                          'kind_id': DISCRETE,
                                          'production_year': CONTINUOUS,
                                          'title': CONTINUOUS, # LIKE
                                          'episode_nr': CONTINUOUS},
                            'keys': {'id', 'kind_id'},},
                'movie_companies': {'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                                    'col_types': {'company_type_id': DISCRETE,
                                                  'company_id': DISCRETE,
                                                  'movie_id': DISCRETE,
                                                  'note': CONTINUOUS}, # LIKE
                                    'keys': {'movie_id', 'company_id', 'company_type_id'},},
                'keyword': {'names': ['id', 'keyword', 'phonetic_code'],
                                    'col_types': {'id': DISCRETE,
                                                  'keyword': CONTINUOUS,}, # LIKE
                                    'keys': {'id'},},
                'movie_keyword': {'names': ['id', 'movie_id', 'keyword_id'],
                                    'col_types': {'keyword_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'keyword_id'},},
                'company_name': {'names': ['id', 'name', 'country_code', 'imdb_id',
                                           'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'country_code': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'info_type': {'names': ['id', 'info'],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE},
                                    'keys': {'id'},},
                'cast_info': {'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                            'col_types': {'role_id': DISCRETE,
                                          'movie_id': DISCRETE,
                                          'person_id': DISCRETE,
                                          'person_role_id': DISCRETE,
                                          'note': CONTINUOUS}, # LIKE
                            'keys': {'movie_id', 'person_id', 'person_role_id', 'role_id'}},
                'movie_info': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                'col_types': {'info_type_id': DISCRETE,
                                              'movie_id': DISCRETE,
                                              'info': CONTINUOUS, # LIKE
                                              'note': CONTINUOUS}, # LIKE
                                'keys': {'movie_id', 'info_type_id'},},
                'name': {'names': ['id', 'name', 'imdb_index', 'imdb_id', 'gender',
                                   'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'gender': CONTINUOUS, # LIKE
                                                  'name': CONTINUOUS, # LIKE
                                                  'name_pcode_cf': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'movie_info_idx': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'info_type_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'info_type_id'},},
                'company_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id'},},
                'kind_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id'},},
                'aka_name': {'names': ['id', 'person_id', 'name', 'imdb_index',
                                       'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'person_id': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'person_id'},},
                'char_name': {'names': ['id', 'name', 'imdb_index', 'imdb_id',
                                        'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'complete_cast': {'names': ['id', 'movie_id', 'subject_id', 'status_id'],
                                    'col_types': {'movie_id': DISCRETE,
                                                  'status_id': DISCRETE,
                                                  'subject_id': DISCRETE},
                                    'keys': {'movie_id', 'status_id', 'subject_id'},},
                'comp_cast_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'role_type': {'names': ['id', 'role'],
                                    'col_types': {'id': DISCRETE,
                                                  'role': DISCRETE},
                                    'keys': {'id'},},
                'link_type': {'names': ['id', 'link'],
                                    'col_types': {'id': DISCRETE,
                                                  'link': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'movie_link': {'names': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
                                    'col_types': {'link_type_id': DISCRETE,
                                                  'movie_id': DISCRETE,
                                                  'linked_movie_id': DISCRETE},
                                    'keys': {'link_type_id', 'movie_id', 'linked_movie_id'},},
                'person_info': {'names': ['id', 'person_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'person_id': DISCRETE,
                                                  'info_type_id': DISCRETE,
                                                  'note': DISCRETE},
                                    'keys': {'person_id', 'info_type_id'},},
                'aka_title': {'names': ['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                                        'production_year', 'phonetic_code', 'episode_of_id',
                                        'season_nr', 'episode_nr', 'note', 'md5sum'],
                                    'col_types': {'movie_id': DISCRETE},
                                    'keys': {'movie_id'},},
                }
    elif experiment == 'job-complex':
        dates = dict()
        intervals = {'title': {'production_year': (1, 2, 4, 8 ,16, 32)}}
        tables = {'title': {'names': ['id', 'title', 'imdb_index', 'kind_id', 'production_year',
                                'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr',
                                'episode_nr', 'series_years', 'md5sum'],
                            'col_types': {'id': DISCRETE,
                                          'kind_id': DISCRETE,
                                          'production_year': CONTINUOUS,
                                          'title': CONTINUOUS, # LIKE
                                          'episode_nr': DISCRETE},
                            'keys': {'id', 'kind_id', 'imdb_index', 'episode_nr'},},
                'movie_companies': {'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                                    'col_types': {'company_type_id': DISCRETE,
                                                  'company_id': DISCRETE,
                                                  'movie_id': DISCRETE,
                                                  'note': CONTINUOUS}, # LIKE
                                    'keys': {'movie_id', 'company_id', 'company_type_id'},},
                'keyword': {'names': ['id', 'keyword', 'phonetic_code'],
                                    'col_types': {'id': DISCRETE,
                                                  'keyword': DISCRETE,
                                                  'phonetic_code': DISCRETE},
                                    'keys': {'id', 'phonetic_code'},},
                'movie_keyword': {'names': ['id', 'movie_id', 'keyword_id'],
                                    'col_types': {'keyword_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'keyword_id'},},
                'company_name': {'names': ['id', 'name', 'country_code', 'imdb_id',
                                           'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'name_pcode_sf': DISCRETE,
                                                  'country_code': DISCRETE,
                                                  'name_pcode_nf': DISCRETE,
                                                  'name': CONTINUOUS}, # LIKE
                                    'keys': {'id', 'name_pcode_sf', 'name_pcode_nf'},},
                'info_type': {'names': ['id', 'info'],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE},
                                    'keys': {'id'},},
                'cast_info': {'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                            'col_types': {'nr_order': DISCRETE,
                                          'movie_id': DISCRETE,
                                          'person_id': DISCRETE,
                                          'person_role_id': DISCRETE,
                                          'note': DISCRETE},
                            'keys': {'movie_id', 'person_id', 'person_role_id', 'nr_order'}},
                'movie_info': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                'col_types': {'info_type_id': DISCRETE,
                                              'movie_id': DISCRETE,
                                              'info': CONTINUOUS, # LIKE
                                              'note': CONTINUOUS}, # LIKE
                                'keys': {'movie_id', 'info_type_id'},},
                'name': {'names': ['id', 'name', 'imdb_index', 'imdb_id', 'gender',
                                   'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'gender': DISCRETE,
                                                  'name': CONTINUOUS,
                                                  'name_pcode_cf': DISCRETE,
                                                  'name_pcode_nf': DISCRETE,
                                                  'surname_pcode': DISCRETE,
                                                  'imdb_index': DISCRETE},
                                    'keys': {'id', 'name', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'imdb_index'},},
                'movie_info_idx': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'info_type_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'info_type_id'},},
                'company_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id'},},
                'kind_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': DISCRETE},
                                    'keys': {'id'},},
                'aka_name': {'names': ['id', 'person_id', 'name', 'imdb_index',
                                       'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'name_pcode_nf': DISCRETE,
                                                  'name_pcode_cf': DISCRETE,
                                                  'imdb_index': DISCRETE,
                                                  'surname_pcode': DISCRETE,},
                                    'keys': {'name_pcode_nf', 'name_pcode_cf', 'imdb_index', 'surname_pcode'},},
                'char_name': {'names': ['id', 'name', 'imdb_index', 'imdb_id',
                                        'name_pcode_nf', 'surname_pcode', 'md5sum'],
                                    'col_types': {'id': DISCRETE,
                                                  'name': CONTINUOUS, # LIKE but also join key
                                                  'surname_pcode': DISCRETE,
                                                  'name_pcode_nf': DISCRETE,}, 
                                    'keys': {'id', 'name_pcode_nf', 'surname_pcode', 'name'},},
                'complete_cast': {'names': ['id', 'movie_id', 'subject_id', 'status_id'],
                                    'col_types': {'movie_id': DISCRETE,
                                                  'status_id': DISCRETE,
                                                  'subject_id': DISCRETE},
                                    'keys': {'movie_id', 'status_id', 'subject_id'},},
                'comp_cast_type': {'names': ['id', 'kind'],
                                    'col_types': {'id': DISCRETE,
                                                  'kind': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'link_type': {'names': ['id', 'link'],
                                    'col_types': {'id': DISCRETE,
                                                  'link': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'movie_link': {'names': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
                                    'col_types': {'link_type_id': DISCRETE,
                                                  'movie_id': DISCRETE,},
                                    'keys': {'link_type_id', 'movie_id'},},
                'aka_title': {'names': ['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                                        'production_year', 'phonetic_code', 'episode_of_id',
                                        'season_nr', 'episode_nr', 'note', 'md5sum'],
                                    'col_types': {'movie_id': DISCRETE,
                                                  'imdb_index': DISCRETE,},
                                    'keys': {'movie_id', 'imdb_index'},},
                }
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    # validate other join keys
    for _, meta in tables.items():
        assert meta['keys'].intersection(meta['col_types']), f"{meta['keys']} must intersect {meta['col_types']}"
        
    return dates, intervals, tables