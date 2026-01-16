def get_config(experiment:str):
    dates = dict()
    intervals = dict()
    tables = dict()

    # note: discrete is intended as non-ordinal
    DISCRETE = 'DISCRETE'
    CONTINUOUS = 'CONTINUOUS'

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
        intervals = {'title': {'production_year': (1, 2, 4, 8 ,16, 32),}}
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
                                          'title': DISCRETE, # LIKE
                                          'episode_nr': CONTINUOUS},
                            'keys': {'id', 'kind_id'},},
                'movie_companies': {'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                                    'col_types': {'company_type_id': DISCRETE,
                                                  'company_id': DISCRETE,
                                                  'movie_id': DISCRETE,
                                                  'note': DISCRETE}, # LIKE
                                    'keys': {'movie_id', 'company_id', 'company_type_id'},},
                'keyword': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'keyword': DISCRETE,}, # LIKE
                                    'keys': {'id'},},
                'movie_keyword': {'names': ['id', 'movie_id', 'keyword_id'],
                                    'col_types': {'keyword_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'keyword_id'},},
                'company_name': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'country_code': DISCRETE,
                                                  'name': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'info_type': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'cast_info': {'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                            'col_types': {'role_id': DISCRETE,
                                          'movie_id': DISCRETE,
                                          'person_id': DISCRETE,
                                          'person_role_id': DISCRETE,
                                          'note': DISCRETE}, # LIKE
                            'keys': {'movie_id', 'person_id', 'person_role_id', 'role_id'}},
                'movie_info': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                'col_types': {'info_type_id': DISCRETE,
                                              'movie_id': DISCRETE,
                                              'info': DISCRETE, # LIKE
                                              'note': DISCRETE}, # LIKE
                                'keys': {'movie_id', 'info_type_id'},},
                'name': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'gender': DISCRETE,
                                                  'name': DISCRETE, # LIKE
                                                  'name_pcode_cf': CONTINUOUS}, # LIKE
                                    'keys': {'id'},},
                'movie_info_idx': {'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                                    'col_types': {'info_type_id': DISCRETE,
                                                  'movie_id': DISCRETE},
                                    'keys': {'movie_id', 'info_type_id'},},
                'company_type': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'kind_type': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'aka_name': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'char_name': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'complete_cast': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'comp_cast_type': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'role_type': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'link_type': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'movie_link': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'person_info': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                'aka_title': {'names': [],
                                    'col_types': {'id': DISCRETE,
                                                  'info': DISCRETE}, # LIKE
                                    'keys': {'id'},},
                }
    else:
        raise 1
    
    # validate other join keys
    for _, meta in tables.items():
        assert meta['keys'].intersection(meta['col_types']), f"{meta['keys']} must intersect {meta['col_types']}"
        
    return dates, intervals, tables