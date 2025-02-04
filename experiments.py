def get_config(experiment:str):
    primary = []
    dates = []
    tables = {}

    # todo: discrete is intended as non-ordinal
    DISCRETE = 'DISCRETE'
    CONTINUOUS = 'CONTINUOUS'

    if experiment == 'stats-ceb':
        primary = {'badges': {'Id'},
                   'posts': {'Id'},
                   'postLinks': {'Id'},
                   'postHistory': {'Id'},
                   'comments': {'Id'},
                   'tags': {'Id'},
                   'users': {'Id'},
                   'votes': {'Id'},}
        dates = {'badges': {'Date',},
                 'comments': {'CreationDate',},
                 'postHistory': {'CreationDate',},
                 'postLinks': {'CreationDate',},
                 'posts': {'CreationDate',},
                 'users': {'CreationDate',},
                 'votes': {'CreationDate',},}
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
    else:
        raise 1
        
    return primary, dates, tables