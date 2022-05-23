#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
from Movies_Model import predict_genre

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Movie Genres Prediction',
    description='Get movies genres based on plot')

ns = api.namespace('predict', 
     description='genres Prediction')
   
parser = api.parser()

parser.add_argument(
    'Plot', 
    type=str, 
    required=True, 
    help="Movie's Plot", 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class GenreApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_genre(args['Plot'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)