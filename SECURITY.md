# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in Transync, please **do not** open a public issue. Instead, please email the maintainers privately with details of the vulnerability.

## Security Considerations

### Model Usage

1. **Input Validation**: Always validate and sanitize user inputs before passing them to the model
2. **Output Verification**: Review translated outputs, especially for critical applications
3. **Model Integrity**: Ensure model weights are downloaded from official sources only
4. **Access Control**: Restrict access to model files if they contain sensitive information

### Deployment

1. **Resource Limits**: Implement rate limiting and resource constraints when deploying as a service
2. **Input Size Limits**: Set reasonable maximum lengths for input texts
3. **Memory Management**: Monitor memory usage during batch processing
4. **Error Handling**: Avoid exposing internal error messages to end users

### Data Privacy

1. **No Data Retention**: The inference script does not store translations by default
2. **Local Processing**: Process data locally when possible to minimize data exposure
3. **User Awareness**: Inform users that translations may be logged if using remote deployment

### Dependencies

- Keep Python dependencies updated: `pip install --upgrade -r requirements.txt`
- Review dependency vulnerabilities periodically
- Use `pip audit` to check for known vulnerabilities: `pip install pip-audit && pip-audit`

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x.x   | ✅ Yes    |

## Python Version Support

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.10+

## Dependencies Security

The following are the primary dependencies. Keep them updated:

- PyTorch:- https://pytorch.org
- Transformers:- https://github.com/huggingface/transformers
- tqdm (optional):- https://github.com/tqdm/tqdm

## Responsible Disclosure

We take security seriously and appreciate your responsible disclosure. When reporting vulnerabilities:

1. Provide detailed information about the vulnerability
2. Include steps to reproduce if applicable
3. Suggest a fix if you have one
4. Allow reasonable time for a fix before public disclosure

## Security Best Practices for Users

1. **Validation**: Test translations thoroughly before deploying to production
2. **Updates**: Keep all dependencies updated
3. **Monitoring**: Log and monitor translation requests
4. **Access**: Restrict access to the model and inference scripts
5. **Backups**: Maintain backups of model files

## Acknowledgments

We appreciate the security research community's efforts to keep open-source software secure. Thank you for helping us maintain Transync's security.